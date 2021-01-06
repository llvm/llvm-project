//===-- CompilerInstance.h - Flang Compiler Instance ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_FLANG_FRONTEND_COMPILERINSTANCE_H
#define LLVM_FLANG_FRONTEND_COMPILERINSTANCE_H

#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/FrontendAction.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Semantics/semantics.h"
#include "llvm/Support/raw_ostream.h"

namespace Fortran::frontend {

class CompilerInstance {

  /// The options used in this compiler instance.
  std::shared_ptr<CompilerInvocation> invocation_;

  /// Flang file  manager.
  std::shared_ptr<Fortran::parser::AllSources> allSources_;

  std::shared_ptr<Fortran::parser::AllCookedSources> allCookedSources_;

  std::shared_ptr<Fortran::parser::Parsing> parsing_;

  /// The stream for diagnostics from Semantics
  llvm::raw_ostream *semaOutputStream_ = &llvm::errs();

  /// The stream for diagnostics from Semantics if owned, otherwise nullptr.
  std::unique_ptr<llvm::raw_ostream> ownedSemaOutputStream_;

  /// The diagnostics engine instance.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diagnostics_;

  /// Holds information about the output file.
  struct OutputFile {
    std::string filename_;
    OutputFile(std::string inputFilename)
        : filename_(std::move(inputFilename)) {}
  };

  /// Output stream that doesn't support seeking (e.g. terminal, pipe).
  /// This stream is normally wrapped in buffer_ostream before being passed
  /// to users (e.g. via CreateOutputFile).
  std::unique_ptr<llvm::raw_fd_ostream> nonSeekStream_;

  /// The list of active output files.
  std::list<OutputFile> outputFiles_;

  /// Holds the output stream provided by the user. Normally, users of
  /// CompilerInstance will call CreateOutputFile to obtain/create an output
  /// stream. If they want to provide their own output stream, this field will
  /// facilitate this. It is optional and will normally be just a nullptr.
  std::unique_ptr<llvm::raw_pwrite_stream> outputStream_;

public:
  explicit CompilerInstance();

  ~CompilerInstance();

  /// @name Compiler Invocation
  /// {

  CompilerInvocation &invocation() {
    assert(invocation_ && "Compiler instance has no invocation!");
    return *invocation_;
  };

  /// Replace the current invocation.
  void set_invocation(std::shared_ptr<CompilerInvocation> value);

  /// }
  /// @name File manager
  /// {

  /// Return the current allSources.
  Fortran::parser::AllSources &allSources() const { return *allSources_; }

  bool HasAllSources() const { return allSources_ != nullptr; }

  parser::AllCookedSources &allCookedSources() {
    assert(allCookedSources_ && "Compiler instance has no AllCookedSources!");
    return *allCookedSources_;
  };

  /// }
  /// @name Parser Operations
  /// {

  /// Return parsing to be used by Actions.
  Fortran::parser::Parsing &parsing() const { return *parsing_; }

  /// }
  /// @name Semantic analysis
  /// {

  /// Replace the current stream for verbose output.
  void set_semaOutputStream(llvm::raw_ostream &Value);

  /// Replace the current stream for verbose output.
  void set_semaOutputStream(std::unique_ptr<llvm::raw_ostream> Value);

  /// Get the current stream for verbose output.
  llvm::raw_ostream &semaOutputStream() { return *semaOutputStream_; }

  /// }
  /// @name High-Level Operations
  /// {

  /// Execute the provided action against the compiler's
  /// CompilerInvocation object.
  /// \param act - The action to execute.
  /// \return - True on success.
  bool ExecuteAction(FrontendAction &act);

  /// }
  /// @name Forwarding Methods
  /// {

  clang::DiagnosticOptions &GetDiagnosticOpts() {
    return invocation_->GetDiagnosticOpts();
  }
  const clang::DiagnosticOptions &GetDiagnosticOpts() const {
    return invocation_->GetDiagnosticOpts();
  }

  FrontendOptions &frontendOpts() { return invocation_->frontendOpts(); }
  const FrontendOptions &frontendOpts() const {
    return invocation_->frontendOpts();
  }

  PreprocessorOptions &preprocessorOpts() {
    return invocation_->preprocessorOpts();
  }
  const PreprocessorOptions &preprocessorOpts() const {
    return invocation_->preprocessorOpts();
  }

  /// }
  /// @name Diagnostics Engine
  /// {

  bool HasDiagnostics() const { return diagnostics_ != nullptr; }

  /// Get the current diagnostics engine.
  clang::DiagnosticsEngine &diagnostics() const {
    assert(diagnostics_ && "Compiler instance has no diagnostics!");
    return *diagnostics_;
  }

  clang::DiagnosticConsumer &GetDiagnosticClient() const {
    assert(diagnostics_ && diagnostics_->getClient() &&
        "Compiler instance has no diagnostic client!");
    return *diagnostics_->getClient();
  }

  /// {
  /// @name Output Files
  /// {

  /// Add an output file onto the list of tracked output files.
  ///
  /// \param outFile - The output file info.
  void AddOutputFile(OutputFile &&outFile);

  /// Clear the output file list.
  void ClearOutputFiles(bool eraseFiles);

  /// Create the default output file (based on the invocation's options) and
  /// add it to the list of tracked output files. If the name of the output
  /// file is not provided, it is derived from the input file.
  ///
  /// \param binary     The mode to open the file in.
  /// \param baseInput  If the invocation contains no output file name (i.e.
  ///                   outputFile_ in FrontendOptions is empty), the input path
  ///                   name to use for deriving the output path.
  /// \param extension  The extension to use for output names derived from
  ///                   \p baseInput.
  /// \return           ostream for the output file or nullptr on error.
  std::unique_ptr<llvm::raw_pwrite_stream> CreateDefaultOutputFile(
      bool binary = true, llvm::StringRef baseInput = "",
      llvm::StringRef extension = "");

  /// Create a new output file
  ///
  /// \param outputPath   The path to the output file.
  /// \param error [out]  On failure, the error.
  /// \param binary       The mode to open the file in.
  /// \return             ostream for the output file or nullptr on error.
  std::unique_ptr<llvm::raw_pwrite_stream> CreateOutputFile(
      llvm::StringRef outputPath, std::error_code &error, bool binary);

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
  static clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> CreateDiagnostics(
      clang::DiagnosticOptions *opts,
      clang::DiagnosticConsumer *client = nullptr, bool shouldOwnClient = true);
  void CreateDiagnostics(
      clang::DiagnosticConsumer *client = nullptr, bool shouldOwnClient = true);

  /// }
  /// @name Output Stream Methods
  /// {
  void set_outputStream(std::unique_ptr<llvm::raw_pwrite_stream> outStream) {
    outputStream_ = std::move(outStream);
  }

  bool IsOutputStreamNull() { return (outputStream_ == nullptr); }

  // Allow the frontend compiler to write in the output stream.
  void WriteOutputStream(const std::string &message) {
    *outputStream_ << message;
  }
};

} // end namespace Fortran::frontend
#endif // LLVM_FLANG_FRONTEND_COMPILERINSTANCE_H
