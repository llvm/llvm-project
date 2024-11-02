//===--- ASTConcept.h - Concepts Related AST Data Structures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include <utility>

namespace clang {
class ConceptDecl;

/// The result of a constraint satisfaction check, containing the necessary
/// information to diagnose an unsatisfied constraint.
class ConstraintSatisfaction : public llvm::FoldingSetNode {
  // The template-like entity that 'owns' the constraint checked here (can be a
  // constrained entity or a concept).
  const NamedDecl *ConstraintOwner = nullptr;
  llvm::SmallVector<TemplateArgument, 4> TemplateArgs;

public:

  ConstraintSatisfaction() = default;

  ConstraintSatisfaction(const NamedDecl *ConstraintOwner,
                         ArrayRef<TemplateArgument> TemplateArgs) :
      ConstraintOwner(ConstraintOwner), TemplateArgs(TemplateArgs.begin(),
                                                     TemplateArgs.end()) { }

  using SubstitutionDiagnostic = std::pair<SourceLocation, StringRef>;
  using Detail = llvm::PointerUnion<Expr *, SubstitutionDiagnostic *>;

  bool IsSatisfied = false;
  bool ContainsErrors = false;

  /// \brief Pairs of unsatisfied atomic constraint expressions along with the
  /// substituted constraint expr, if the template arguments could be
  /// substituted into them, or a diagnostic if substitution resulted in an
  /// invalid expression.
  llvm::SmallVector<std::pair<const Expr *, Detail>, 4> Details;

  void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &C) {
    Profile(ID, C, ConstraintOwner, TemplateArgs);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, const ASTContext &C,
                      const NamedDecl *ConstraintOwner,
                      ArrayRef<TemplateArgument> TemplateArgs);
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
  bool ContainsErrors : 1;

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

/// \brief Common data class for constructs that reference concepts with
/// template arguments.
class ConceptReference {
protected:
  // \brief The optional nested name specifier used when naming the concept.
  NestedNameSpecifierLoc NestedNameSpec;

  /// \brief The location of the template keyword, if specified when naming the
  /// concept.
  SourceLocation TemplateKWLoc;

  /// \brief The concept name used.
  DeclarationNameInfo ConceptName;

  /// \brief The declaration found by name lookup when the expression was
  /// created.
  /// Can differ from NamedConcept when, for example, the concept was found
  /// through a UsingShadowDecl.
  NamedDecl *FoundDecl;

  /// \brief The concept named.
  ConceptDecl *NamedConcept;

  /// \brief The template argument list source info used to specialize the
  /// concept.
  const ASTTemplateArgumentListInfo *ArgsAsWritten;

public:
  ConceptReference(NestedNameSpecifierLoc NNS, SourceLocation TemplateKWLoc,
                   DeclarationNameInfo ConceptNameInfo, NamedDecl *FoundDecl,
                   ConceptDecl *NamedConcept,
                   const ASTTemplateArgumentListInfo *ArgsAsWritten)
      : NestedNameSpec(NNS), TemplateKWLoc(TemplateKWLoc),
        ConceptName(ConceptNameInfo), FoundDecl(FoundDecl),
        NamedConcept(NamedConcept), ArgsAsWritten(ArgsAsWritten) {}

  ConceptReference()
      : FoundDecl(nullptr), NamedConcept(nullptr), ArgsAsWritten(nullptr) {}

  const NestedNameSpecifierLoc &getNestedNameSpecifierLoc() const {
    return NestedNameSpec;
  }

  const DeclarationNameInfo &getConceptNameInfo() const { return ConceptName; }

  SourceLocation getConceptNameLoc() const {
    return getConceptNameInfo().getLoc();
  }

  SourceLocation getTemplateKWLoc() const { return TemplateKWLoc; }

  NamedDecl *getFoundDecl() const {
    return FoundDecl;
  }

  ConceptDecl *getNamedConcept() const {
    return NamedConcept;
  }

  const ASTTemplateArgumentListInfo *getTemplateArgsAsWritten() const {
    return ArgsAsWritten;
  }

  /// \brief Whether or not template arguments were explicitly specified in the
  /// concept reference (they might not be in type constraints, for example)
  bool hasExplicitTemplateArgs() const {
    return ArgsAsWritten != nullptr;
  }
};

class TypeConstraint : public ConceptReference {
  /// \brief The immediately-declared constraint expression introduced by this
  /// type-constraint.
  Expr *ImmediatelyDeclaredConstraint = nullptr;

public:
  TypeConstraint(NestedNameSpecifierLoc NNS,
                 DeclarationNameInfo ConceptNameInfo, NamedDecl *FoundDecl,
                 ConceptDecl *NamedConcept,
                 const ASTTemplateArgumentListInfo *ArgsAsWritten,
                 Expr *ImmediatelyDeclaredConstraint) :
      ConceptReference(NNS, /*TemplateKWLoc=*/SourceLocation(), ConceptNameInfo,
                       FoundDecl, NamedConcept, ArgsAsWritten),
      ImmediatelyDeclaredConstraint(ImmediatelyDeclaredConstraint) {}

  /// \brief Get the immediately-declared constraint expression introduced by
  /// this type-constraint, that is - the constraint expression that is added to
  /// the associated constraints of the enclosing declaration in practice.
  Expr *getImmediatelyDeclaredConstraint() const {
    return ImmediatelyDeclaredConstraint;
  }

  void print(llvm::raw_ostream &OS, PrintingPolicy Policy) const;
};

} // clang

#endif // LLVM_CLANG_AST_ASTCONCEPT_H
