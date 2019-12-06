//===--- ASTConcept.h - Concepts Related AST Data Structures ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides AST data structures related to concepts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTCONCEPT_H
#define LLVM_CLANG_AST_ASTCONCEPT_H
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <utility>
namespace clang {

/// \brief The result of a constraint satisfaction check, containing the
/// necessary information to diagnose an unsatisfied constraint.
struct ConstraintSatisfaction {
  using SubstitutionDiagnostic = std::pair<SourceLocation, StringRef>;
  using Detail = llvm::PointerUnion<Expr *, SubstitutionDiagnostic *>;

  bool IsSatisfied = false;

  /// \brief Pairs of unsatisfied atomic constraint expressions along with the
  /// substituted constraint expr, if the template arguments could be
  /// substituted into them, or a diagnostic if substitution resulted in an
  /// invalid expression.
  llvm::SmallVector<std::pair<const Expr *, Detail>, 4> Details;

  // This can leak if used in an AST node, use ASTConstraintSatisfaction
  // instead.
  void *operator new(size_t bytes, ASTContext &C) = delete;
};

/// Pairs of unsatisfied atomic constraint expressions along with the
/// substituted constraint expr, if the template arguments could be
/// substituted into them, or a diagnostic if substitution resulted in
/// an invalid expression.
using UnsatisfiedConstraintRecord =
    std::pair<const Expr *,
              llvm::PointerUnion<Expr *,
                                 std::pair<SourceLocation, StringRef> *>>;

/// \brief The result of a constraint satisfaction check, containing the
/// necessary information to diagnose an unsatisfied constraint.
///
/// This is safe to store in an AST node, as opposed to ConstraintSatisfaction.
struct ASTConstraintSatisfaction final :
    llvm::TrailingObjects<ASTConstraintSatisfaction,
                          UnsatisfiedConstraintRecord> {
  std::size_t NumRecords;
  bool IsSatisfied : 1;

  const UnsatisfiedConstraintRecord *begin() const {
    return getTrailingObjects<UnsatisfiedConstraintRecord>();
  }

  const UnsatisfiedConstraintRecord *end() const {
    return getTrailingObjects<UnsatisfiedConstraintRecord>() + NumRecords;
  }

  ASTConstraintSatisfaction(const ASTContext &C,
                            const ConstraintSatisfaction &Satisfaction);

  static ASTConstraintSatisfaction *
  Create(const ASTContext &C, const ConstraintSatisfaction &Satisfaction);
};

} // clang

#endif // LLVM_CLANG_AST_ASTCONCEPT_H