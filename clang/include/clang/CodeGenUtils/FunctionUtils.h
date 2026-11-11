//===--- FunctionUtils.h - Shared function emission queries -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file holds the queries and checks that both classic CodeGen and CIR
// CodeGen need while emitting a function body.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H
#define LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace clang {
class ASTContext;
class CallExpr;
class DiagnosticsEngine;
class FunctionDecl;
} // namespace clang

namespace clang::CodeGenUtils {

/// Decide whether we need to emit the lifetime markers.
bool shouldEmitLifetimeMarkers(const CodeGenOptions &CGOpts,
                               const LangOptions &LangOpts);

/// The target features that always_inline functions inherit from the
/// always_inline functions they call.
///
/// An always_inline function that is only emitted where it is used has no
/// code of its own that would have to run without the target features of a
/// callee. Calling an always_inline function that needs more target features
/// than it was declared with is therefore not an error. It then needs those
/// features as well, as though it had been declared with them, and it is the
/// functions it gets inlined into that have to provide them.
class InheritedTargetFeatures {
public:
  /// Whether \p FD inherits the target features that its callees need.
  static bool canInherit(ASTContext &Ctx, const FunctionDecl *FD);

  /// Whether no function has inherited target features.
  bool empty() const { return Inherited.empty(); }

  /// Whether \p FD has inherited target features.
  bool contains(const FunctionDecl *FD) const;

  /// Enables the target features that \p FD has inherited in \p FeatureMap.
  void addTo(ASTContext &Ctx, const FunctionDecl *FD,
             llvm::StringMap<bool> &FeatureMap) const;

  /// Records that \p Caller calls the always_inline function \p Callee at
  /// \p Loc, and diagnoses the target features that the functions which the
  /// code of \p Callee ends up in lack.
  void addCall(ASTContext &Ctx, DiagnosticsEngine &Diags, SourceLocation Loc,
               const FunctionDecl *Caller, const FunctionDecl *Callee);

private:
  struct CallSite {
    const FunctionDecl *Caller;
    SourceLocation Loc;
  };

  /// A call to an always_inline function whose target features have to be
  /// provided by whichever function its code ends up in.
  struct InlinedCall {
    const FunctionDecl *Callee;
    CallSite Site;
  };

  void require(ASTContext &Ctx, DiagnosticsEngine &Diags,
               const FunctionDecl *Host, SourceLocation Loc, InlinedCall Call);

  /// The calls to each always_inline function. They are needed once its body
  /// shows which target features it inherits, which is usually after its
  /// callers were emitted.
  llvm::DenseMap<const FunctionDecl *, llvm::SmallVector<CallSite, 1>> CallsTo;

  /// The calls whose callees each function inherits target features from.
  llvm::DenseMap<const FunctionDecl *, llvm::SmallVector<InlinedCall, 1>>
      Inherited;
};

/// Check that a call to a target-specific builtin has the required target
/// features enabled in the caller, emitting an error diagnostic if not.
/// \p Caller is the FunctionDecl of the enclosing function (may be null).
/// Without \p Inherited, no function inherits target features.
void checkTargetFeatures(ASTContext &Ctx, DiagnosticsEngine &Diags,
                         const LangOptions &LangOpts, const CallExpr *E,
                         const FunctionDecl *Caller,
                         const FunctionDecl *TargetDecl,
                         InheritedTargetFeatures *Inherited = nullptr);

/// Overload taking a raw source location instead of a CallExpr.
void checkTargetFeatures(ASTContext &Ctx, DiagnosticsEngine &Diags,
                         const LangOptions &LangOpts, SourceLocation Loc,
                         const FunctionDecl *Caller,
                         const FunctionDecl *TargetDecl,
                         InheritedTargetFeatures *Inherited = nullptr);

} // namespace clang::CodeGenUtils

#endif // LLVM_CLANG_CODEGENUTILS_FUNCTIONUTILS_H
