//===--- CompilerInstance.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Semantics/semantics.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran::frontend;

CompilerInstance::CompilerInstance()
    : invocation(new CompilerInvocation()),
      allSources(new Fortran::parser::AllSources()),
      allCookedSources(new Fortran::parser::AllCookedSources(*allSources)),
      parsing(new Fortran::parser::Parsing(*allCookedSources)) {
  // TODO: This is a good default during development, but ultimately we should
  // give the user the opportunity to specify this.
  allSources->set_encoding(Fortran::parser::Encoding::UTF_8);
}

CompilerInstance::~CompilerInstance() {
  assert(outputFiles.empty() && "Still output files in flight?");
}

void CompilerInstance::setInvocation(
    std::shared_ptr<CompilerInvocation> value) {
  invocation = std::move(value);
}

void CompilerInstance::setSemaOutputStream(raw_ostream &value) {
  ownedSemaOutputStream.release();
  semaOutputStream = &value;
}

void CompilerInstance::setSemaOutputStream(std::unique_ptr<raw_ostream> value) {
  ownedSemaOutputStream.swap(value);
  semaOutputStream = ownedSemaOutputStream.get();
}

// Helper method to generate the path of the output file. The following logic
// applies:
// 1. If the user specifies the output file via `-o`, then use that (i.e.
//    the outputFilename parameter).
// 2. If the user does not specify the name of the output file, derive it from
//    the input file (i.e. inputFilename + extension)
// 3. If the output file is not specified and the input file is `-`, then set
//    the output file to `-` as well.
static std::string getOutputFilePath(llvm::StringRef outputFilename,
                                     llvm::StringRef inputFilename,
                                     llvm::StringRef extension) {

  // Output filename _is_ specified. Just use that.
  if (!outputFilename.empty())
    return std::string(outputFilename);

  // Output filename _is not_ specified. Derive it from the input file name.
  std::string outFile = "-";
  if (!extension.empty() && (inputFilename != "-")) {
    llvm::SmallString<128> path(inputFilename);
    llvm::sys::path::replace_extension(path, extension);
    outFile = std::string(path.str());
  }

  return outFile;
}

std::unique_ptr<llvm::raw_pwrite_stream>
CompilerInstance::createDefaultOutputFile(bool binary, llvm::StringRef baseName,
                                          llvm::StringRef extension) {

  // Get the path of the output file
  std::string outputFilePath =
      getOutputFilePath(getFrontendOpts().outputFile, baseName, extension);

  // Create the output file
  llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>> os =
      createOutputFileImpl(outputFilePath, binary);

  // If successful, add the file to the list of tracked output files and
  // return.
  if (os) {
    outputFiles.emplace_back(OutputFile(outputFilePath));
    return std::move(*os);
  }

  // If unsuccessful, issue an error and return Null
  unsigned diagID = getDiagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "unable to open output file '%0': '%1'");
  getDiagnostics().Report(diagID)
      << outputFilePath << llvm::errorToErrorCode(os.takeError()).message();
  return nullptr;
}

llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>>
CompilerInstance::createOutputFileImpl(llvm::StringRef outputFilePath,
                                       bool binary) {

  // Creates the file descriptor for the output file
  std::unique_ptr<llvm::raw_fd_ostream> os;

  std::error_code error;
  os.reset(new llvm::raw_fd_ostream(
      outputFilePath, error,
      (binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_TextWithCRLF)));
  if (error) {
    return llvm::errorCodeToError(error);
  }

  // For seekable streams, just return the stream corresponding to the output
  // file.
  if (!binary || os->supportsSeeking())
    return std::move(os);

  // For non-seekable streams, we need to wrap the output stream into something
  // that supports 'pwrite' and takes care of the ownership for us.
  return std::make_unique<llvm::buffer_unique_ostream>(std::move(os));
}

void CompilerInstance::clearOutputFiles(bool eraseFiles) {
  for (OutputFile &of : outputFiles)
    if (!of.filename.empty() && eraseFiles)
      llvm::sys::fs::remove(of.filename);

  outputFiles.clear();
}

bool CompilerInstance::executeAction(FrontendAction &act) {
  auto &invoc = this->getInvocation();

  llvm::Triple targetTriple{llvm::Triple(invoc.getTargetOpts().triple)};
  if (targetTriple.getArch() == llvm::Triple::ArchType::x86_64) {
    invoc.getDefaultKinds().set_quadPrecisionKind(10);
  }

  // Set some sane defaults for the frontend.
  invoc.setDefaultFortranOpts();
  // Update the fortran options based on user-based input.
  invoc.setFortranOpts();
  // Set the encoding to read all input files in based on user input.
  allSources->set_encoding(invoc.getFortranOpts().encoding);
  // Create the semantics context and set semantic options.
  invoc.setSemanticsOpts(*this->allCookedSources);
  // Set options controlling lowering to FIR.
  invoc.setLoweringOptions();

  // Run the frontend action `act` for every input file.
  for (const FrontendInputFile &fif : getFrontendOpts().inputs) {
    if (act.beginSourceFile(*this, fif)) {
      if (llvm::Error err = act.execute()) {
        consumeError(std::move(err));
      }
      act.endSourceFile();
    }
  }
  return !getDiagnostics().getClient()->getNumErrors();
}

void CompilerInstance::createDiagnostics(clang::DiagnosticConsumer *client,
                                         bool shouldOwnClient) {
  diagnostics =
      createDiagnostics(&getDiagnosticOpts(), client, shouldOwnClient);
}

clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine>
CompilerInstance::createDiagnostics(clang::DiagnosticOptions *opts,
                                    clang::DiagnosticConsumer *client,
                                    bool shouldOwnClient) {
  clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
      new clang::DiagnosticIDs());
  clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags(
      new clang::DiagnosticsEngine(diagID, opts));

  // Create the diagnostic client for reporting errors or for
  // implementing -verify.
  if (client) {
    diags->setClient(client, shouldOwnClient);
  } else {
    diags->setClient(new TextDiagnosticPrinter(llvm::errs(), opts));
  }
  return diags;
}
