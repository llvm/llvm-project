//===-- CompilerInstance.h - Flang Compiler Instance ------------*- C++ -*-===//
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

#ifndef FORTRAN_FRONTEND_COMPILERINSTANCE_H
#define FORTRAN_FRONTEND_COMPILERINSTANCE_H

#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendAction.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/semantics.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

namespace Fortran::frontend {

/// Helper class for managing a single instance of the Flang compiler.
///
/// This class serves two purposes:
///  (1) It manages the various objects which are necessary to run the compiler
///  (2) It provides utility routines for constructing and manipulating the
///      common Flang objects.
///
/// The compiler instance generally owns the instance of all the objects that it
/// manages. However, clients can still share objects by manually setting the
/// object and retaking ownership prior to destroying the CompilerInstance.
///
/// The compiler instance is intended to simplify clients, but not to lock them
/// in to the compiler instance for everything. When possible, utility functions
/// come in two forms; a short form that reuses the CompilerInstance objects,
/// and a long form that takes explicit instances of any required objects.
class CompilerInstance {

  /// The options used in this compiler instance.
  std::shared_ptr<CompilerInvocation> invocation;

  /// Flang file  manager.
  std::shared_ptr<Fortran::parser::AllSources> allSources;

  std::shared_ptr<Fortran::parser::AllCookedSources> allCookedSources;

  std::shared_ptr<Fortran::parser::Parsing> parsing;

  std::unique_ptr<Fortran::semantics::Semantics> semantics;

  std::unique_ptr<Fortran::semantics::RuntimeDerivedTypeTables> rtTyTables;

  std::unique_ptr<Fortran::semantics::SemanticsContext> semaContext;

  std::unique_ptr<llvm::TargetMachine> targetMachine;

  /// The stream for diagnostics from Semantics
  llvm::raw_ostream *semaOutputStream = &llvm::errs();

  /// The stream for diagnostics from Semantics if owned, otherwise nullptr.
  std::unique_ptr<llvm::raw_ostream> ownedSemaOutputStream;

  /// The diagnostics engine instance.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diagnostics;

  /// Holds information about the output file.
  struct OutputFile {
    std::string filename;
    OutputFile(std::string inputFilename)
        : filename(std::move(inputFilename)) {}
  };

  /// The list of active output files.
  std::list<OutputFile> outputFiles;

  /// Holds the output stream provided by the user. Normally, users of
  /// CompilerInstance will call CreateOutputFile to obtain/create an output
  /// stream. If they want to provide their own output stream, this field will
  /// facilitate this. It is optional and will normally be just a nullptr.
  std::unique_ptr<llvm::raw_pwrite_stream> outputStream;

public:
  explicit CompilerInstance();

  ~CompilerInstance();

  /// @name Compiler Invocation
  /// {

  CompilerInvocation &getInvocation() {
    assert(invocation && "Compiler instance has no invocation!");
    return *invocation;
  };

  /// Replace the current invocation.
  void setInvocation(std::shared_ptr<CompilerInvocation> value);

  /// }
  /// @name File manager
  /// {

  /// Return the current allSources.
  Fortran::parser::AllSources &getAllSources() const { return *allSources; }

  bool hasAllSources() const { return allSources != nullptr; }

  parser::AllCookedSources &getAllCookedSources() {
    assert(allCookedSources && "Compiler instance has no AllCookedSources!");
    return *allCookedSources;
  };

  /// }
  /// @name Parser Operations
  /// {

  /// Return parsing to be used by Actions.
  Fortran::parser::Parsing &getParsing() const { return *parsing; }

  /// }
  /// @name Semantic analysis
  /// {

  Fortran::semantics::SemanticsContext &getSemanticsContext() {
    return *semaContext;
  }
  const Fortran::semantics::SemanticsContext &getSemanticsContext() const {
    return *semaContext;
  }

  /// Replace the current stream for verbose output.
  void setSemaOutputStream(llvm::raw_ostream &value);

  /// Replace the current stream for verbose output.
  void setSemaOutputStream(std::unique_ptr<llvm::raw_ostream> value);

  /// Get the current stream for verbose output.
  llvm::raw_ostream &getSemaOutputStream() { return *semaOutputStream; }

  Fortran::semantics::Semantics &getSemantics() { return *semantics; }
  const Fortran::semantics::Semantics &getSemantics() const {
    return *semantics;
  }

  void setSemantics(std::unique_ptr<Fortran::semantics::Semantics> sema) {
    semantics = std::move(sema);
  }

  void setRtTyTables(
      std::unique_ptr<Fortran::semantics::RuntimeDerivedTypeTables> tables) {
    rtTyTables = std::move(tables);
  }

  Fortran::semantics::RuntimeDerivedTypeTables &getRtTyTables() {
    assert(rtTyTables && "Missing runtime derived type tables!");
    return *rtTyTables;
  }

  /// }
  /// @name High-Level Operations
  /// {

  /// Execute the provided action against the compiler's
  /// CompilerInvocation object.
  /// \param act - The action to execute.
  /// \return - True on success.
  bool executeAction(FrontendAction &act);

  /// }
  /// @name Forwarding Methods
  /// {

  clang::DiagnosticOptions &getDiagnosticOpts() {
    return invocation->getDiagnosticOpts();
  }
  const clang::DiagnosticOptions &getDiagnosticOpts() const {
    return invocation->getDiagnosticOpts();
  }

  FrontendOptions &getFrontendOpts() { return invocation->getFrontendOpts(); }
  const FrontendOptions &getFrontendOpts() const {
    return invocation->getFrontendOpts();
  }

  PreprocessorOptions &preprocessorOpts() {
    return invocation->getPreprocessorOpts();
  }
  const PreprocessorOptions &preprocessorOpts() const {
    return invocation->getPreprocessorOpts();
  }

  /// }
  /// @name Diagnostics Engine
  /// {

  bool hasDiagnostics() const { return diagnostics != nullptr; }

  /// Get the current diagnostics engine.
  clang::DiagnosticsEngine &getDiagnostics() const {
    assert(diagnostics && "Compiler instance has no diagnostics!");
    return *diagnostics;
  }

  clang::DiagnosticConsumer &getDiagnosticClient() const {
    assert(diagnostics && diagnostics->getClient() &&
           "Compiler instance has no diagnostic client!");
    return *diagnostics->getClient();
  }

  /// {
  /// @name Output Files
  /// {

  /// Clear the output file list.
  void clearOutputFiles(bool eraseFiles);

  /// Create the default output file (based on the invocation's options) and
  /// add it to the list of tracked output files. If the name of the output
  /// file is not provided, it will be derived from the input file.
  ///
  /// \param binary     The mode to open the file in.
  /// \param baseInput  If the invocation contains no output file name (i.e.
  ///                   outputFile in FrontendOptions is empty), the input path
  ///                   name to use for deriving the output path.
  /// \param extension  The extension to use for output names derived from
  ///                   \p baseInput.
  /// \return           Null on error, ostream for the output file otherwise
  std::unique_ptr<llvm::raw_pwrite_stream>
  createDefaultOutputFile(bool binary = true, llvm::StringRef baseInput = "",
                          llvm::StringRef extension = "");

  /// {
  /// @name Target Machine
  /// {

  /// Get the target machine.
  const llvm::TargetMachine &getTargetMachine() const {
    assert(targetMachine && "target machine was not set");
    return *targetMachine;
  }
  llvm::TargetMachine &getTargetMachine() {
    assert(targetMachine && "target machine was not set");
    return *targetMachine;
  }

  /// Sets up LLVM's TargetMachine.
  bool setUpTargetMachine();

  /// Produces the string which represents target feature
  std::string getTargetFeatures();

private:
  /// Create a new output file
  ///
  /// \param outputPath   The path to the output file.
  /// \param binary       The mode to open the file in.
  /// \return             Null on error, ostream for the output file otherwise
  llvm::Expected<std::unique_ptr<llvm::raw_pwrite_stream>>
  createOutputFileImpl(llvm::StringRef outputPath, bool binary);

public:
  /// }
  /// @name Construction Utility Methods
  /// {

  /// Create a DiagnosticsEngine object
  ///
  /// If no diagnostic client is provided, this method creates a
  /// DiagnosticConsumer that is owned by the returned diagnostic object. If
  /// using directly the caller is responsible for releasing the returned
  /// DiagnosticsEngine's client eventually.
  ///
  /// \param opts - The diagnostic options; note that the created text
  /// diagnostic object contains a reference to these options.
  ///
  /// \param client - If non-NULL, a diagnostic client that will be attached to
  /// (and optionally, depending on /p shouldOwnClient, owned by) the returned
  /// DiagnosticsEngine object.
  ///
  /// \return The new object on success, or null on failure.
  static clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine>
  createDiagnostics(clang::DiagnosticOptions *opts,
                    clang::DiagnosticConsumer *client = nullptr,
                    bool shouldOwnClient = true);
  void createDiagnostics(clang::DiagnosticConsumer *client = nullptr,
                         bool shouldOwnClient = true);

  /// }
  /// @name Output Stream Methods
  /// {
  void setOutputStream(std::unique_ptr<llvm::raw_pwrite_stream> outStream) {
    outputStream = std::move(outStream);
  }

  bool isOutputStreamNull() { return (outputStream == nullptr); }

  // Allow the frontend compiler to write in the output stream.
  void writeOutputStream(const std::string &message) {
    *outputStream << message;
  }

  /// Get the user specified output stream.
  llvm::raw_pwrite_stream &getOutputStream() {
    assert(outputStream &&
           "Compiler instance has no user-specified output stream!");
    return *outputStream;
  }
};

} // end namespace Fortran::frontend
#endif // FORTRAN_FRONTEND_COMPILERINSTANCE_H
