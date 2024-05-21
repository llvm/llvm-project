//===--- SemaExceptionSpec.h --- C++ exception specification ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file declares routines for C++ exception specification testing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAEXCEPTIONSPEC_H
#define LLVM_CLANG_SEMA_SEMAEXCEPTIONSPEC_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ExceptionSpecificationType.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <utility>

namespace clang {
class SemaExceptionSpec : public SemaBase {
public:
  SemaExceptionSpec(Sema &S);

  /// All the overriding functions seen during a class definition
  /// that had their exception spec checks delayed, plus the overridden
  /// function.
  SmallVector<std::pair<const CXXMethodDecl *, const CXXMethodDecl *>, 2>
      DelayedOverridingExceptionSpecChecks;

  /// All the function redeclarations seen during a class definition that had
  /// their exception spec checks delayed, plus the prior declaration they
  /// should be checked against. Except during error recovery, the new decl
  /// should always be a friend declaration, as that's the only valid way to
  /// redeclare a special member before its class is complete.
  SmallVector<std::pair<FunctionDecl *, FunctionDecl *>, 2>
      DelayedEquivalentExceptionSpecChecks;

  /// Determine if we're in a case where we need to (incorrectly) eagerly
  /// parse an exception specification to work around a libstdc++ bug.
  bool isLibstdcxxEagerExceptionSpecHack(const Declarator &D);

  /// Check the given noexcept-specifier, convert its expression, and compute
  /// the appropriate ExceptionSpecificationType.
  ExprResult ActOnNoexceptSpec(Expr *NoexceptExpr,
                               ExceptionSpecificationType &EST);

  CanThrowResult canThrow(const Stmt *E);
  /// Determine whether the callee of a particular function call can throw.
  /// E, D and Loc are all optional.
  static CanThrowResult canCalleeThrow(Sema &S, const Expr *E, const Decl *D,
                                       SourceLocation Loc = SourceLocation());
  const FunctionProtoType *ResolveExceptionSpec(SourceLocation Loc,
                                                const FunctionProtoType *FPT);
  void UpdateExceptionSpec(FunctionDecl *FD,
                           const FunctionProtoType::ExceptionSpecInfo &ESI);
  bool CheckSpecifiedExceptionType(QualType &T, SourceRange Range);
  bool CheckDistantExceptionSpec(QualType T);
  bool CheckEquivalentExceptionSpec(FunctionDecl *Old, FunctionDecl *New);
  bool CheckEquivalentExceptionSpec(const FunctionProtoType *Old,
                                    SourceLocation OldLoc,
                                    const FunctionProtoType *New,
                                    SourceLocation NewLoc);
  bool CheckEquivalentExceptionSpec(const PartialDiagnostic &DiagID,
                                    const PartialDiagnostic &NoteID,
                                    const FunctionProtoType *Old,
                                    SourceLocation OldLoc,
                                    const FunctionProtoType *New,
                                    SourceLocation NewLoc);
  bool handlerCanCatch(QualType HandlerType, QualType ExceptionType);
  bool CheckExceptionSpecSubset(
      const PartialDiagnostic &DiagID, const PartialDiagnostic &NestedDiagID,
      const PartialDiagnostic &NoteID, const PartialDiagnostic &NoThrowDiagID,
      const FunctionProtoType *Superset, bool SkipSupersetFirstParameter,
      SourceLocation SuperLoc, const FunctionProtoType *Subset,
      bool SkipSubsetFirstParameter, SourceLocation SubLoc);
  bool CheckParamExceptionSpec(
      const PartialDiagnostic &NestedDiagID, const PartialDiagnostic &NoteID,
      const FunctionProtoType *Target, bool SkipTargetFirstParameter,
      SourceLocation TargetLoc, const FunctionProtoType *Source,
      bool SkipSourceFirstParameter, SourceLocation SourceLoc);

  bool CheckExceptionSpecCompatibility(Expr *From, QualType ToType);

  /// CheckOverridingFunctionExceptionSpec - Checks whether the exception
  /// spec is a subset of base spec.
  bool CheckOverridingFunctionExceptionSpec(const CXXMethodDecl *New,
                                            const CXXMethodDecl *Old);

  void CheckDelayedMemberExceptionSpecs();

  /// Helper class that collects exception specifications for
  /// implicitly-declared special member functions.
  class ImplicitExceptionSpecification {
    // Pointer to allow copying
    Sema *Self;
    // We order exception specifications thus:
    // noexcept is the most restrictive, but is only used in C++11.
    // throw() comes next.
    // Then a throw(collected exceptions)
    // Finally no specification, which is expressed as noexcept(false).
    // throw(...) is used instead if any called function uses it.
    ExceptionSpecificationType ComputedEST;
    llvm::SmallPtrSet<CanQualType, 4> ExceptionsSeen;
    SmallVector<QualType, 4> Exceptions;

    void ClearExceptions() {
      ExceptionsSeen.clear();
      Exceptions.clear();
    }

  public:
    explicit ImplicitExceptionSpecification(Sema &Self);

    /// Get the computed exception specification type.
    ExceptionSpecificationType getExceptionSpecType() const {
      assert(!isComputedNoexcept(ComputedEST) &&
             "noexcept(expr) should not be a possible result");
      return ComputedEST;
    }

    /// The number of exceptions in the exception specification.
    unsigned size() const { return Exceptions.size(); }

    /// The set of exceptions in the exception specification.
    const QualType *data() const { return Exceptions.data(); }

    /// Integrate another called method into the collected data.
    void CalledDecl(SourceLocation CallLoc, const CXXMethodDecl *Method);

    /// Integrate an invoked expression into the collected data.
    void CalledExpr(Expr *E) { CalledStmt(E); }

    /// Integrate an invoked statement into the collected data.
    void CalledStmt(Stmt *S);

    /// Overwrite an EPI's exception specification with this
    /// computed exception specification.
    FunctionProtoType::ExceptionSpecInfo getExceptionSpec() const;
  };

  /// Check the given exception-specification and update the
  /// exception specification information with the results.
  void checkExceptionSpecification(bool IsTopLevel,
                                   ExceptionSpecificationType EST,
                                   ArrayRef<ParsedType> DynamicExceptions,
                                   ArrayRef<SourceRange> DynamicExceptionRanges,
                                   Expr *NoexceptExpr,
                                   SmallVectorImpl<QualType> &Exceptions,
                                   FunctionProtoType::ExceptionSpecInfo &ESI);

  /// Add an exception-specification to the given member or friend function
  /// (or function template). The exception-specification was parsed
  /// after the function itself was declared.
  void actOnDelayedExceptionSpecification(
      Decl *D, ExceptionSpecificationType EST, SourceRange SpecificationRange,
      ArrayRef<ParsedType> DynamicExceptions,
      ArrayRef<SourceRange> DynamicExceptionRanges, Expr *NoexceptExpr);

  /// Build an exception spec for destructors that don't have one.
  ///
  /// C++11 says that user-defined destructors with no exception spec get one
  /// that looks as if the destructor was implicitly declared.
  void AdjustDestructorExceptionSpec(CXXDestructorDecl *Destructor);

  /// Mark the exception specifications of all virtual member functions
  /// in the given class as needed.
  void MarkVirtualMemberExceptionSpecsNeeded(SourceLocation Loc,
                                             const CXXRecordDecl *RD);

  void MergeVarDeclExceptionSpecs(VarDecl *New, VarDecl *Old);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAEXCEPTIONSPEC_H