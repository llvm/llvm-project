//===--- InlineConceptRequirement.cpp ----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ParsedAST.h"
#include "SourceCode.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprConcepts.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace clangd {
namespace {
/// Inlines a concept requirement.
///
/// Before:
///   template <typename T> void f(T) requires foo<T> {}
///                                            ^^^^^^
/// After:
///   template <foo T> void f(T) {}
class InlineConceptRequirement : public Tweak {
public:
  const char *id() const final;

  auto prepare(const Selection &Inputs) -> bool override;
  auto apply(const Selection &Inputs) -> Expected<Effect> override;
  auto title() const -> std::string override {
    return "Inline concept requirement";
  }
  auto kind() const -> llvm::StringLiteral override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  const ConceptSpecializationExpr *ConceptSpecializationExpression;
  const TemplateTypeParmDecl *TemplateTypeParameterDeclaration;
  const syntax::Token *RequiresToken;

  static auto getTemplateParameterIndexOfTemplateArgument(
      const TemplateArgument &TemplateArgument) -> std::optional<int>;
  auto generateRequiresReplacement(ASTContext &)
      -> llvm::Expected<tooling::Replacement>;
  auto generateRequiresTokenReplacement(const syntax::TokenBuffer &)
      -> tooling::Replacement;
  auto generateTemplateParameterReplacement(ASTContext &Context)
      -> llvm::Expected<tooling::Replacement>;

  static auto findToken(const ParsedAST *, const SourceRange &,
                        const tok::TokenKind) -> const syntax::Token *;

  template <typename T, typename NodeKind>
  static auto findNode(const SelectionTree::Node &Root)
      -> std::tuple<const T *, const SelectionTree::Node *>;

  template <typename T>
  static auto findExpression(const SelectionTree::Node &Root)
      -> std::tuple<const T *, const SelectionTree::Node *> {
    return findNode<T, Expr>(Root);
  }

  template <typename T>
  static auto findDeclaration(const SelectionTree::Node &Root)
      -> std::tuple<const T *, const SelectionTree::Node *> {
    return findNode<T, Decl>(Root);
  }
};

REGISTER_TWEAK(InlineConceptRequirement)

auto InlineConceptRequirement::prepare(const Selection &Inputs) -> bool {
  // Check if C++ version is 20 or higher
  if (!Inputs.AST->getLangOpts().CPlusPlus20)
    return false;

  const auto *Root = Inputs.ASTSelection.commonAncestor();
  if (!Root)
    return false;

  const SelectionTree::Node *ConceptSpecializationExpressionTreeNode;
  std::tie(ConceptSpecializationExpression,
           ConceptSpecializationExpressionTreeNode) =
      findExpression<ConceptSpecializationExpr>(*Root);
  if (!ConceptSpecializationExpression)
    return false;

  // Only allow concepts that are direct children of function template
  // declarations or function declarations. This excludes conjunctions of
  // concepts which are not handled.
  const auto *ParentDeclaration =
      ConceptSpecializationExpressionTreeNode->Parent->ASTNode.get<Decl>();
  if (!isa_and_nonnull<FunctionTemplateDecl>(ParentDeclaration) &&
      !isa_and_nonnull<FunctionDecl>(ParentDeclaration))
    return false;

  const FunctionTemplateDecl *FunctionTemplateDeclaration =
      std::get<0>(findDeclaration<FunctionTemplateDecl>(*Root));
  if (!FunctionTemplateDeclaration)
    return false;

  auto TemplateArguments =
      ConceptSpecializationExpression->getTemplateArguments();
  if (TemplateArguments.size() != 1)
    return false;

  auto TemplateParameterIndex =
      getTemplateParameterIndexOfTemplateArgument(TemplateArguments[0]);
  if (!TemplateParameterIndex)
    return false;

  TemplateTypeParameterDeclaration = dyn_cast_or_null<TemplateTypeParmDecl>(
      FunctionTemplateDeclaration->getTemplateParameters()->getParam(
          *TemplateParameterIndex));
  if (!TemplateTypeParameterDeclaration->wasDeclaredWithTypename())
    return false;

  RequiresToken =
      findToken(Inputs.AST, FunctionTemplateDeclaration->getSourceRange(),
                tok::kw_requires);
  if (!RequiresToken)
    return false;

  return true;
}

auto InlineConceptRequirement::apply(const Selection &Inputs)
    -> Expected<Tweak::Effect> {
  auto &Context = Inputs.AST->getASTContext();
  auto &TokenBuffer = Inputs.AST->getTokens();

  tooling::Replacements Replacements{};

  auto TemplateParameterReplacement =
      generateTemplateParameterReplacement(Context);

  if (auto Err = TemplateParameterReplacement.takeError())
    return Err;

  if (auto Err = Replacements.add(*TemplateParameterReplacement))
    return Err;

  auto RequiresReplacement = generateRequiresReplacement(Context);

  if (auto Err = RequiresReplacement.takeError())
    return Err;

  if (auto Err = Replacements.add(*RequiresReplacement))
    return Err;

  if (auto Err =
          Replacements.add(generateRequiresTokenReplacement(TokenBuffer)))
    return Err;

  return Effect::mainFileEdit(Context.getSourceManager(), Replacements);
}

auto InlineConceptRequirement::getTemplateParameterIndexOfTemplateArgument(
    const TemplateArgument &TemplateArgument) -> std::optional<int> {
  if (TemplateArgument.getKind() != TemplateArgument.Type)
    return {};

  auto TemplateArgumentType = TemplateArgument.getAsType();
  if (!TemplateArgumentType->isTemplateTypeParmType())
    return {};

  const auto *TemplateTypeParameterType =
      TemplateArgumentType->getAs<TemplateTypeParmType>();
  if (!TemplateTypeParameterType)
    return {};

  return TemplateTypeParameterType->getIndex();
}

auto InlineConceptRequirement::generateRequiresReplacement(ASTContext &Context)
    -> llvm::Expected<tooling::Replacement> {
  auto &SourceManager = Context.getSourceManager();

  auto RequiresRange =
      toHalfOpenFileRange(SourceManager, Context.getLangOpts(),
                          ConceptSpecializationExpression->getSourceRange());
  if (!RequiresRange)
    return error("Could not obtain range of the 'requires' branch. Macros?");

  return tooling::Replacement(
      SourceManager, CharSourceRange::getCharRange(*RequiresRange), "");
}

auto InlineConceptRequirement::generateRequiresTokenReplacement(
    const syntax::TokenBuffer &TokenBuffer) -> tooling::Replacement {
  auto &SourceManager = TokenBuffer.sourceManager();

  auto Spelling =
      TokenBuffer.spelledForExpanded(llvm::ArrayRef(*RequiresToken));

  auto DeletionRange =
      syntax::Token::range(SourceManager, Spelling->front(), Spelling->back())
          .toCharRange(SourceManager);

  return tooling::Replacement(SourceManager, DeletionRange, "");
}

auto InlineConceptRequirement::generateTemplateParameterReplacement(
    ASTContext &Context) -> llvm::Expected<tooling::Replacement> {
  auto &SourceManager = Context.getSourceManager();

  auto ConceptName = ConceptSpecializationExpression->getNamedConcept()
                         ->getQualifiedNameAsString();

  auto TemplateParameterName =
      TemplateTypeParameterDeclaration->getQualifiedNameAsString();

  auto TemplateParameterReplacement = ConceptName + ' ' + TemplateParameterName;

  auto TemplateParameterRange =
      toHalfOpenFileRange(SourceManager, Context.getLangOpts(),
                          TemplateTypeParameterDeclaration->getSourceRange());

  if (!TemplateParameterRange)
    return error("Could not obtain range of the template parameter. Macros?");

  return tooling::Replacement(
      SourceManager, CharSourceRange::getCharRange(*TemplateParameterRange),
      TemplateParameterReplacement);
}

auto clang::clangd::InlineConceptRequirement::findToken(
    const ParsedAST *AST, const SourceRange &SourceRange,
    const tok::TokenKind TokenKind) -> const syntax::Token * {
  auto &TokenBuffer = AST->getTokens();
  const auto &Tokens = TokenBuffer.expandedTokens(SourceRange);

  const auto Predicate = [TokenKind](const auto &Token) {
    return Token.kind() == TokenKind;
  };

  auto It = std::find_if(Tokens.begin(), Tokens.end(), Predicate);

  if (It == Tokens.end())
    return nullptr;

  return It;
}

template <typename T, typename NodeKind>
auto InlineConceptRequirement::findNode(const SelectionTree::Node &Root)
    -> std::tuple<const T *, const SelectionTree::Node *> {

  for (const auto *Node = &Root; Node; Node = Node->Parent) {
    if (const T *Result = dyn_cast_or_null<T>(Node->ASTNode.get<NodeKind>()))
      return {Result, Node};
  }

  return {};
}

} // namespace
} // namespace clangd
} // namespace clang
