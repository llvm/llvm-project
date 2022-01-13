//===--- ExpandAutoType.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "refactor/Tweak.h"

#include "XRefs.h"
#include "support/Logger.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include <AST.h>
#include <climits>
#include <memory>
#include <string>

namespace clang {
namespace clangd {
namespace {

/// Expand the "auto" type to the derived type
/// Before:
///    auto x = Something();
///    ^^^^
/// After:
///    MyClass x = Something();
///    ^^^^^^^
/// FIXME: Handle decltype as well
class ExpandAutoType : public Tweak {
public:
  const char *id() const final;
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }
  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;
  std::string title() const override;

private:
  /// Cache the AutoTypeLoc, so that we do not need to search twice.
  llvm::Optional<clang::AutoTypeLoc> CachedLocation;
};

REGISTER_TWEAK(ExpandAutoType)

std::string ExpandAutoType::title() const { return "Expand auto type"; }

// Structured bindings must use auto, e.g. `const auto& [a,b,c] = ...;`.
// Return whether N (an AutoTypeLoc) is such an auto that must not be expanded.
bool isStructuredBindingType(const SelectionTree::Node *N) {
  // Walk up the TypeLoc chain, because auto may be qualified.
  while (N && N->ASTNode.get<TypeLoc>())
    N = N->Parent;
  // The relevant type is the only direct type child of a Decomposition.
  return N && N->ASTNode.get<DecompositionDecl>();
}

// Returns true iff Node is a lambda, and thus should not be expanded. Loc is
// the location of the auto type.
bool isDeducedAsLambda(const SelectionTree::Node *Node, SourceLocation Loc) {
  // getDeducedType() does a traversal, which we want to avoid in prepare().
  // But at least check this isn't auto x = []{...};, which can't ever be
  // expanded.
  // (It would be nice if we had an efficient getDeducedType(), instead).
  for (const auto *It = Node; It; It = It->Parent) {
    if (const auto *DD = It->ASTNode.get<DeclaratorDecl>()) {
      if (DD->getTypeSourceInfo() &&
          DD->getTypeSourceInfo()->getTypeLoc().getBeginLoc() == Loc) {
        if (auto *RD = DD->getType()->getAsRecordDecl())
          return RD->isLambda();
      }
    }
  }
  return false;
}

// Returns true iff "auto" in Node is really part of the template parameter,
// which we cannot expand.
bool isTemplateParam(const SelectionTree::Node *Node) {
  if (Node->Parent)
    if (Node->Parent->ASTNode.get<NonTypeTemplateParmDecl>())
      return true;
  return false;
}

bool ExpandAutoType::prepare(const Selection& Inputs) {
  CachedLocation = llvm::None;
  if (auto *Node = Inputs.ASTSelection.commonAncestor()) {
    if (auto *TypeNode = Node->ASTNode.get<TypeLoc>()) {
      if (const AutoTypeLoc Result = TypeNode->getAs<AutoTypeLoc>()) {
        if (!isStructuredBindingType(Node) &&
            !isDeducedAsLambda(Node, Result.getBeginLoc()) &&
            !isTemplateParam(Node))
          CachedLocation = Result;
      }
    }
  }

  return (bool) CachedLocation;
}

Expected<Tweak::Effect> ExpandAutoType::apply(const Selection& Inputs) {
  auto &SrcMgr = Inputs.AST->getSourceManager();

  llvm::Optional<clang::QualType> DeducedType = getDeducedType(
      Inputs.AST->getASTContext(), CachedLocation->getBeginLoc());

  // if we can't resolve the type, return an error message
  if (DeducedType == llvm::None || (*DeducedType)->isUndeducedAutoType())
    return error("Could not deduce type for 'auto' type");

  // if it's a lambda expression, return an error message
  if (isa<RecordType>(*DeducedType) &&
      cast<RecordType>(*DeducedType)->getDecl()->isLambda()) {
    return error("Could not expand type of lambda expression");
  }

  // if it's a function expression, return an error message
  // naively replacing 'auto' with the type will break declarations.
  // FIXME: there are other types that have similar problems
  if (DeducedType->getTypePtr()->isFunctionPointerType()) {
    return error("Could not expand type of function pointer");
  }

  std::string PrettyTypeName = printType(*DeducedType,
      Inputs.ASTSelection.commonAncestor()->getDeclContext());

  tooling::Replacement
      Expansion(SrcMgr, CharSourceRange(CachedLocation->getSourceRange(), true),
                PrettyTypeName);

  return Effect::mainFileEdit(SrcMgr, tooling::Replacements(Expansion));
}

} // namespace
} // namespace clangd
} // namespace clang
