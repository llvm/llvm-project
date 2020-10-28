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

#include <cassert>
#include <memory>

namespace Fortran::frontend {

class CompilerInstance {

  /// The options used in this compiler instance.
  std::shared_ptr<CompilerInvocation> invocation_;

  /// The diagnostics engine instance.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> diagnostics_;

public:
  explicit CompilerInstance();

  ~CompilerInstance();
  CompilerInvocation &GetInvocation() {
    assert(invocation_ && "Compiler instance has no invocation!");
    return *invocation_;
  };

  /// }
  /// @name Forwarding Methods
  /// {

  clang::DiagnosticOptions &GetDiagnosticOpts() {
    return invocation_->GetDiagnosticOpts();
  }
  const clang::DiagnosticOptions &GetDiagnosticOpts() const {
    return invocation_->GetDiagnosticOpts();
  }

  FrontendOptions &GetFrontendOpts() { return invocation_->GetFrontendOpts(); }
  const FrontendOptions &GetFrontendOpts() const {
    return invocation_->GetFrontendOpts();
  }

  /// }
  /// @name Diagnostics Engine
  /// {

  bool HasDiagnostics() const { return diagnostics_ != nullptr; }

  /// Get the current diagnostics engine.
  clang::DiagnosticsEngine &GetDiagnostics() const {
    assert(diagnostics_ && "Compiler instance has no diagnostics!");
    return *diagnostics_;
  }

  /// SetDiagnostics - Replace the current diagnostics engine.
  void SetDiagnostics(clang::DiagnosticsEngine *value);

  clang::DiagnosticConsumer &GetDiagnosticClient() const {
    assert(diagnostics_ && diagnostics_->getClient() &&
        "Compiler instance has no diagnostic client!");
    return *diagnostics_->getClient();
  }

  /// Get the current diagnostics engine.
  clang::DiagnosticsEngine &getDiagnostics() const {
    assert(diagnostics_ && "Compiler instance has no diagnostics!");
    return *diagnostics_;
  }

  /// }
  /// @name Construction Utility Methods
  /// {

  /// Create a DiagnosticsEngine object with a the TextDiagnosticPrinter.
  ///
  /// If no diagnostic client is provided, this creates a
  /// DiagnosticConsumer that is owned by the returned diagnostic
  /// object, if using directly the caller is responsible for
  /// releasing the returned DiagnosticsEngine's client eventually.
  ///
  /// \param opts - The diagnostic options; note that the created text
  /// diagnostic object contains a reference to these options.
  ///
  /// \param client If non-NULL, a diagnostic client that will be
  /// attached to (and, then, owned by) the returned DiagnosticsEngine
  /// object.
  ///
  /// \return The new object on success, or null on failure.
  static clang::IntrusiveRefCntPtr<clang::DiagnosticsEngine> CreateDiagnostics(
      clang::DiagnosticOptions *opts,
      clang::DiagnosticConsumer *client = nullptr, bool shouldOwnClient = true);
  void CreateDiagnostics(
      clang::DiagnosticConsumer *client = nullptr, bool shouldOwnClient = true);
};

} // end namespace Fortran::frontend
#endif // LLVM_FLANG_FRONTEND_COMPILERINSTANCE_H
