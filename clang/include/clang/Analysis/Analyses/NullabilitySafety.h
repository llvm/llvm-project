//===- NullabilitySafety.h - Nullability safety analysis --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines APIs for invoking flow-sensitive nullability analysis
// that detects dereferences of nullable pointers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_NULLABILITYSAFETY_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_NULLABILITYSAFETY_H

#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace clang {

class AnalysisDeclContext;
class Decl;
class Expr;
class FieldDecl;
class FunctionDecl;
class LangOptions;
class ParmVarDecl;
class TranslationUnitDecl;
class VarDecl;

enum class NullabilityEvidence { Nonnull, Nullable, MaybeNull, Unknown };

/// Receives the diagnostics and evidence produced by
/// runNullabilitySafetyAnalysis. Only handleNullableDereference is required;
/// the rest default to no-ops. Evidence feeds the NullabilitySafety SSAF
/// summary extractor.
class NullabilitySafetyHandler {
public:
  virtual ~NullabilitySafetyHandler();
  /// A pointer that may be null is dereferenced (*p, p->f, p[i], or a smart
  /// pointer's operator* / operator->).
  virtual void handleNullableDereference(const Expr *DerefExpr,
                                         QualType PtrType) = 0;
  /// A pointer that may be null is used in arithmetic (p + i, p++, p += i).
  /// \p VD identifies which pointer is at fault; `a - b` reports both operands
  /// at the same location, so the type alone cannot tell them apart.
  virtual void handleNullableArithmetic(const Expr *ArithExpr, QualType PtrType,
                                        const VarDecl *VD) {}
  /// A function with a _Nonnull return type returns a value that may be null.
  virtual void handleNullableReturn(const Expr *ReturnExpr) {}
  /// A _Nonnull variable is initialized or assigned a value that may be null.
  virtual void handleNullableAssignment(const Expr *AssignExpr,
                                        const VarDecl *LHSVar) {}
  /// A _Nonnull field is assigned or aggregate-initialized with a value that
  /// may be null.
  virtual void handleNullableMemberAssignment(const Expr *AssignExpr,
                                              const FieldDecl *Member) {}
  /// A value that may be null is passed to a _Nonnull (or
  /// __attribute__((nonnull))) parameter.
  virtual void handleNullableArgument(const Expr *ArgExpr,
                                      const ParmVarDecl *Param) {}

  /// A _Nonnull smart pointer member of this object is null on every path
  /// to the end of a method (reset, released, or moved from).
  virtual void handleNullableMemberAtExit(SourceLocation Loc,
                                          const FieldDecl *Member) {}

  /// Evidence for annotation inference: a pointer member is assigned, a
  /// function returns a pointer, or a pointer argument is passed.
  virtual void handleMemberAssignEvidence(const Expr *AssignExpr,
                                          const FieldDecl *Member,
                                          NullabilityEvidence Kind) {}
  virtual void handleReturnEvidence(const Expr *RetExpr,
                                    const FunctionDecl *Func,
                                    NullabilityEvidence Kind) {}
  virtual void handleParameterEvidence(const Expr *ArgExpr,
                                       const ParmVarDecl *Param,
                                       const FunctionDecl *Func,
                                       NullabilityEvidence Kind) {}

  /// Called after the fixpoint when every return in the function yields a
  /// provably non-null value (&x, this, new, a narrowed variable, ...), so
  /// callers can treat the result as implicitly _Nonnull.
  virtual void handleAllReturnsNonnull(const FunctionDecl *Func) {}

  /// Called before and after the callbacks for each function that
  /// runNullabilitySafetyOnTU analyzes.
  virtual void startFunction(const Decl *Def) {}
  virtual void finishFunction(const Decl *Def) {}
};

/// Facts about other functions, consulted while analyzing one function.
class NullabilitySafetySummaries {
public:
  virtual ~NullabilitySafetySummaries();
  /// Whether every return of \p Func is known to be non-null. Used to narrow
  /// the result of a call to \p Func.
  virtual bool isKnownAllReturnsNonnull(const FunctionDecl *Func) const = 0;
};

struct NullabilitySafetyOptions {
  /// How an unannotated (_Null_unspecified) pointer is treated.
  NullabilityKind DefaultNullability = NullabilityKind::Unspecified;
  /// Enables the built-in list of C library functions that return null on
  /// failure (malloc, fopen, ...).
  bool LibcNullableReturns = true;

  static NullabilitySafetyOptions fromLangOptions(const LangOptions &LO);
};

/// Run the flow-sensitive nullability analysis over the CFG of the function
/// in \p AC, reporting through \p Handler. \p Summaries may be null.
void runNullabilitySafetyAnalysis(AnalysisDeclContext &AC,
                                  NullabilitySafetyHandler &Handler,
                                  const NullabilitySafetyOptions &Options,
                                  const NullabilitySafetySummaries *Summaries);

bool hasExplicitNullabilityAnnotations(const Decl *D);

/// Whether \p D is checked: every function when a default nullability is
/// set, otherwise only functions with explicit nullability annotations.
bool isNullabilitySafetyOptedIn(const Decl *D,
                                const NullabilitySafetyOptions &Options);

const Decl *getNullabilitySafetyDefinition(const Decl *D);

/// Analyze every opted-in function definition in \p TU that \p Filter
/// accepts, callees before callers, so a caller sees which callees return
/// non-null on every path. No inference happens inside recursive cycles.
void runNullabilitySafetyOnTU(TranslationUnitDecl *TU,
                              NullabilitySafetyHandler &Handler,
                              const NullabilitySafetyOptions &Options,
                              llvm::function_ref<bool(const Decl *)> Filter);

} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_NULLABILITYSAFETY_H
