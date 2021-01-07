//===--- CompilerInstance.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Semantics/semantics.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran::frontend;

CompilerInstance::CompilerInstance()
    : invocation_(new CompilerInvocation()),
      allSources_(new Fortran::parser::AllSources()),
      allCookedSources_(new Fortran::parser::AllCookedSources(*allSources_)),
      parsing_(new Fortran::parser::Parsing(*allCookedSources_)) {

  // TODO: This is a good default during development, but ultimately we should
  // give the user the opportunity to specify this.
  allSources_->set_encoding(Fortran::parser::Encoding::UTF_8);
}

CompilerInstance::~CompilerInstance() {
  assert(outputFiles_.empty() && "Still output files in flight?");
}

void CompilerInstance::set_invocation(
    std::shared_ptr<CompilerInvocation> value) {
  invocation_ = std::move(value);
}

void CompilerInstance::set_semaOutputStream(raw_ostream &Value) {
  ownedSemaOutputStream_.release();
  semaOutputStream_ = &Value;
}

void CompilerInstance::set_semaOutputStream(
    std::unique_ptr<raw_ostream> Value) {
  ownedSemaOutputStream_.swap(Value);
  semaOutputStream_ = ownedSemaOutputStream_.get();
}

void CompilerInstance::AddOutputFile(OutputFile &&outFile) {
  outputFiles_.push_back(std::move(outFile));
}

// Helper method to generate the path of the output file. The following logic
// applies:
// 1. If the user specifies the output file via `-o`, then use that (i.e.
//    the outputFilename parameter).
// 2. If the user does not specify the name of the output file, derive it from
//    the input file (i.e. inputFilename + extension)
// 3. If the output file is not specified and the input file is `-`, then set
//    the output file to `-` as well.
static std::string GetOutputFilePath(llvm::StringRef outputFilename,
    llvm::StringRef inputFilename, llvm::StringRef extension) {

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
CompilerInstance::CreateDefaultOutputFile(
    bool binary, llvm::StringRef baseName, llvm::StringRef extension) {
  std::string outputPathName;
  std::error_code ec;

  // Get the path of the output file
  std::string outputFilePath =
      GetOutputFilePath(frontendOpts().outputFile_, baseName, extension);

  // Create the output file
  std::unique_ptr<llvm::raw_pwrite_stream> os =
      CreateOutputFile(outputFilePath, ec, binary);

  // Add the file to the list of tracked output files (provided it was created
  // successfully)
  if (os)
    AddOutputFile(OutputFile(outputPathName));

  return os;
}

std::unique_ptr<llvm::raw_pwrite_stream> CompilerInstance::CreateOutputFile(
    llvm::StringRef outputFilePath, std::error_code &error, bool binary) {

  // Creates the file descriptor for the output file
  std::unique_ptr<llvm::raw_fd_ostream> os;
  std::string osFile;
  if (!os) {
    osFile = outputFilePath;
    os.reset(new llvm::raw_fd_ostream(osFile, error,
        (binary ? llvm::sys::fs::OF_None : llvm::sys::fs::OF_Text)));
    if (error)
      return nullptr;
  }

  // Return the stream corresponding to the output file.
  // For non-seekable streams, wrap it in llvm::buffer_ostream first.
  if (!binary || os->supportsSeeking())
    return std::move(os);

  assert(!nonSeekStream_ && "The non-seek stream has already been set!");
  auto b = std::make_unique<llvm::buffer_ostream>(*os);
  nonSeekStream_ = std::move(os);
  return std::move(b);
}

void CompilerInstance::ClearOutputFiles(bool eraseFiles) {
  for (OutputFile &of : outputFiles_)
    if (!of.filename_.empty() && eraseFiles)
      llvm::sys::fs::remove(of.filename_);

  outputFiles_.clear();
  nonSeekStream_.reset();
}

bool CompilerInstance::ExecuteAction(FrontendAction &act) {
  // Set some sane defaults for the frontend.
  // TODO: Instead of defaults we should be setting these options based on the
  // user input.
  this->invocation().SetDefaultFortranOpts();
  // Set the fortran options to user-based input.
  this->invocation().setFortranOpts();

  // Connect Input to a CompileInstance
  for (const FrontendInputFile &fif : frontendOpts().inputs_) {
    if (act.BeginSourceFile(*this, fif)) {
      if (llvm::Error err = act.Execute()) {
        consumeError(std::move(err));
      }
      act.EndSourceFile();
    }
  }
  return !diagnostics().getClient()->getNumErrors();
}

void CompilerInstance::CreateDiagnostics(
    clang::DiagnosticConsumer *client, bool shouldOwnClient) {
  diagnostics_ =
      CreateDiagnostics(&GetDiagnosticOpts(), client, shouldOwnClient);
}

clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine>
CompilerInstance::CreateDiagnostics(clang::DiagnosticOptions *opts,
    clang::DiagnosticConsumer *client, bool shouldOwnClient) {
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
