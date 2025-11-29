//===-- AbbreviateFunctionTemplate.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "XRefs.h"
#include "refactor/Tweak.h"
#include "support/Logger.h"
#include "clang/AST/ASTContext.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <numeric>

namespace clang {
namespace clangd {
namespace {
/// Converts a function template to its abbreviated form using auto parameters.
/// Before:
///     template <std::integral T>
///     auto foo(T param) { }
///          ^^^^^^^^^^^
/// After:
///     auto foo(std::integral auto param) { }
class AbbreviateFunctionTemplate : public Tweak {
public:
  const char *id() const final;

  bool prepare(const Selection &Inputs) override;
  Expected<Effect> apply(const Selection &Inputs) override;

  std::string title() const override {
    return llvm::formatv("Abbreviate function template");
  }

  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  static const char *AutoKeywordSpelling;
  const FunctionTemplateDecl *FunctionTemplateDeclaration;

  struct TemplateParameterInfo {
    const TypeConstraint *Constraint;
    unsigned int FunctionParameterIndex;
    llvm::SmallVector<tok::TokenKind> FunctionParameterQualifiers;
    llvm::SmallVector<tok::TokenKind> FunctionParameterTypeQualifiers;
  };

  llvm::SmallVector<TemplateParameterInfo> TemplateParameterInfoList;

  bool traverseFunctionParameters(size_t NumberOfTemplateParameters);

  llvm::Expected<tooling::Replacements>
  generateFunctionParameterReplacements(const ASTContext &Context);

  llvm::Expected<tooling::Replacement> generateFunctionParameterReplacement(
      const TemplateParameterInfo &TemplateParameterInfo,
      const ASTContext &Context);

  llvm::Expected<tooling::Replacement>
  generateTemplateDeclarationReplacement(const ASTContext &Context);

  static std::tuple<QualType, llvm::SmallVector<tok::TokenKind>,
                    llvm::SmallVector<tok::TokenKind>>
  deconstructType(QualType Type);
};

REGISTER_TWEAK(AbbreviateFunctionTemplate)

const char *AbbreviateFunctionTemplate::AutoKeywordSpelling =
    getKeywordSpelling(tok::kw_auto);

template <typename T>
const T *findDeclaration(const SelectionTree::Node &Root) {
  for (const auto *Node = &Root; Node; Node = Node->Parent) {
    if (const T *Result = dyn_cast_or_null<T>(Node->ASTNode.get<Decl>()))
      return Result;
  }

  return nullptr;
}

const char *getSpellingForQualifier(tok::TokenKind const &Qualifier) {
  if (const auto *Spelling = getKeywordSpelling(Qualifier))
    return Spelling;

  if (const auto *Spelling = getPunctuatorSpelling(Qualifier))
    return Spelling;

  return nullptr;
}

bool AbbreviateFunctionTemplate::prepare(const Selection &Inputs) {
  const auto *CommonAncestor = Inputs.ASTSelection.commonAncestor();
  if (!CommonAncestor)
    return false;

  FunctionTemplateDeclaration =
      findDeclaration<FunctionTemplateDecl>(*CommonAncestor);

  if (!FunctionTemplateDeclaration)
    return false;

  auto *TemplateParameters =
      FunctionTemplateDeclaration->getTemplateParameters();

  auto NumberOfTemplateParameters = TemplateParameters->size();
  TemplateParameterInfoList =
      llvm::SmallVector<TemplateParameterInfo>(NumberOfTemplateParameters);

  // Check how many times each template parameter is referenced.
  // Depending on the number of references it can be checked
  // if the refactoring is possible:
  // - exactly one: The template parameter was declared but never used, which
  //                means we know for sure it doesn't appear as a parameter.
  // - exactly two: The template parameter was used exactly once, either as a
  //                parameter or somewhere else. This is the case we are
  //                interested in.
  // - more than two: The template parameter was either used for multiple
  //                  parameters or somewhere else in the function.
  for (unsigned TemplateParameterIndex = 0;
       TemplateParameterIndex < NumberOfTemplateParameters;
       TemplateParameterIndex++) {
    auto *TemplateParameter =
        TemplateParameters->getParam(TemplateParameterIndex);
    auto *TemplateParameterInfo =
        &TemplateParameterInfoList[TemplateParameterIndex];

    auto *TemplateParameterDeclaration =
        dyn_cast_or_null<TemplateTypeParmDecl>(TemplateParameter);
    if (!TemplateParameterDeclaration)
      return false;

    TemplateParameterInfo->Constraint =
        TemplateParameterDeclaration->getTypeConstraint();

    auto TemplateParameterPosition = sourceLocToPosition(
        Inputs.AST->getSourceManager(), TemplateParameter->getEndLoc());

    auto FindReferencesLimit = 3;
    auto ReferencesResult =
        findReferences(*Inputs.AST, TemplateParameterPosition,
                       FindReferencesLimit, Inputs.Index);

    if (ReferencesResult.References.size() != 2)
      return false;
  }

  return traverseFunctionParameters(NumberOfTemplateParameters);
}

Expected<Tweak::Effect>
AbbreviateFunctionTemplate::apply(const Selection &Inputs) {
  auto &Context = Inputs.AST->getASTContext();
  auto FunctionParameterReplacements =
      generateFunctionParameterReplacements(Context);

  if (auto Err = FunctionParameterReplacements.takeError())
    return Err;

  auto Replacements = *FunctionParameterReplacements;
  auto TemplateDeclarationReplacement =
      generateTemplateDeclarationReplacement(Context);

  if (auto Err = TemplateDeclarationReplacement.takeError())
    return Err;

  if (auto Err = Replacements.add(*TemplateDeclarationReplacement))
    return Err;

  return Effect::mainFileEdit(Context.getSourceManager(), Replacements);
}

bool AbbreviateFunctionTemplate::traverseFunctionParameters(
    size_t NumberOfTemplateParameters) {
  auto CurrentTemplateParameterBeingChecked = 0u;
  auto FunctionParameters =
      FunctionTemplateDeclaration->getAsFunction()->parameters();

  for (auto ParameterIndex = 0u; ParameterIndex < FunctionParameters.size();
       ParameterIndex++) {
    auto [RawType, ParameterTypeQualifiers, ParameterQualifiers] =
        deconstructType(FunctionParameters[ParameterIndex]->getOriginalType());

    if (!RawType->isTemplateTypeParmType())
      continue;

    auto TemplateParameterIndex =
        dyn_cast<TemplateTypeParmType>(RawType)->getIndex();

    if (TemplateParameterIndex != CurrentTemplateParameterBeingChecked)
      return false;

    auto *TemplateParameterInfo =
        &TemplateParameterInfoList[TemplateParameterIndex];
    TemplateParameterInfo->FunctionParameterIndex = ParameterIndex;
    TemplateParameterInfo->FunctionParameterTypeQualifiers =
        ParameterTypeQualifiers;
    TemplateParameterInfo->FunctionParameterQualifiers = ParameterQualifiers;

    CurrentTemplateParameterBeingChecked++;
  }

  // All defined template parameters need to be used as function parameters
  return CurrentTemplateParameterBeingChecked == NumberOfTemplateParameters;
}

llvm::Expected<tooling::Replacements>
AbbreviateFunctionTemplate::generateFunctionParameterReplacements(
    const ASTContext &Context) {
  tooling::Replacements Replacements;
  for (const auto &TemplateParameterInfo : TemplateParameterInfoList) {
    auto FunctionParameterReplacement =
        generateFunctionParameterReplacement(TemplateParameterInfo, Context);

    if (auto Err = FunctionParameterReplacement.takeError())
      return Err;

    if (auto Err = Replacements.add(*FunctionParameterReplacement))
      return Err;
  }

  return Replacements;
}

llvm::Expected<tooling::Replacement>
AbbreviateFunctionTemplate::generateFunctionParameterReplacement(
    const TemplateParameterInfo &TemplateParameterInfo,
    const ASTContext &Context) {
  auto &SourceManager = Context.getSourceManager();

  const auto *Function = FunctionTemplateDeclaration->getAsFunction();
  auto *Parameter =
      Function->getParamDecl(TemplateParameterInfo.FunctionParameterIndex);
  auto ParameterName = Parameter->getDeclName().getAsString();

  llvm::SmallVector<std::string> ParameterTokens{};

  if (const auto *TypeConstraint = TemplateParameterInfo.Constraint) {
    auto *ConceptReference = TypeConstraint->getConceptReference();
    auto *NamedConcept = ConceptReference->getNamedConcept();

    ParameterTokens.push_back(NamedConcept->getQualifiedNameAsString());

    if (const auto *TemplateArgs = TypeConstraint->getTemplateArgsAsWritten()) {
      auto TemplateArgsRange = SourceRange(TemplateArgs->getLAngleLoc(),
                                           TemplateArgs->getRAngleLoc());
      auto TemplateArgsSource = toSourceCode(SourceManager, TemplateArgsRange);
      ParameterTokens.push_back(TemplateArgsSource.str() + '>');
    }
  }

  ParameterTokens.push_back(AutoKeywordSpelling);

  for (const auto &Qualifier :
       TemplateParameterInfo.FunctionParameterTypeQualifiers) {
    ParameterTokens.push_back(getSpellingForQualifier(Qualifier));
  }

  ParameterTokens.push_back(ParameterName);

  for (const auto &Qualifier :
       TemplateParameterInfo.FunctionParameterQualifiers) {
    ParameterTokens.push_back(getSpellingForQualifier(Qualifier));
  }

  auto FunctionTypeReplacementText = std::accumulate(
      ParameterTokens.begin(), ParameterTokens.end(), std::string{},
      [](auto Result, auto Token) { return std::move(Result) + " " + Token; });

  auto FunctionParameterRange = toHalfOpenFileRange(
      SourceManager, Context.getLangOpts(), Parameter->getSourceRange());

  if (!FunctionParameterRange)
    return error("Could not obtain range of the template parameter. Macros?");

  return tooling::Replacement(
      SourceManager, CharSourceRange::getCharRange(*FunctionParameterRange),
      FunctionTypeReplacementText);
}

llvm::Expected<tooling::Replacement>
AbbreviateFunctionTemplate::generateTemplateDeclarationReplacement(
    const ASTContext &Context) {
  auto &SourceManager = Context.getSourceManager();
  auto *TemplateParameters =
      FunctionTemplateDeclaration->getTemplateParameters();

  auto TemplateDeclarationRange =
      toHalfOpenFileRange(SourceManager, Context.getLangOpts(),
                          TemplateParameters->getSourceRange());

  if (!TemplateDeclarationRange)
    return error("Could not obtain range of the template parameter. Macros?");

  auto CharRange = CharSourceRange::getCharRange(*TemplateDeclarationRange);
  return tooling::Replacement(SourceManager, CharRange, "");
}

std::tuple<QualType, llvm::SmallVector<tok::TokenKind>,
           llvm::SmallVector<tok::TokenKind>>
AbbreviateFunctionTemplate::deconstructType(QualType Type) {
  llvm::SmallVector<tok::TokenKind> ParameterTypeQualifiers{};
  llvm::SmallVector<tok::TokenKind> ParameterQualifiers{};

  if (Type->isIncompleteArrayType()) {
    ParameterQualifiers.push_back(tok::l_square);
    ParameterQualifiers.push_back(tok::r_square);
    Type = Type->castAsArrayTypeUnsafe()->getElementType();
  }

  if (isa<PackExpansionType>(Type))
    ParameterTypeQualifiers.push_back(tok::ellipsis);

  Type = Type.getNonPackExpansionType();

  if (Type->isRValueReferenceType()) {
    ParameterTypeQualifiers.push_back(tok::ampamp);
    Type = Type.getNonReferenceType();
  }

  if (Type->isLValueReferenceType()) {
    ParameterTypeQualifiers.push_back(tok::amp);
    Type = Type.getNonReferenceType();
  }

  if (Type.isConstQualified()) {
    ParameterTypeQualifiers.push_back(tok::kw_const);
  }

  while (Type->isPointerType()) {
    ParameterTypeQualifiers.push_back(tok::star);
    Type = Type->getPointeeType();

    if (Type.isConstQualified()) {
      ParameterTypeQualifiers.push_back(tok::kw_const);
    }
  }

  std::reverse(ParameterTypeQualifiers.begin(), ParameterTypeQualifiers.end());

  return {Type, ParameterTypeQualifiers, ParameterQualifiers};
}

} // namespace
} // namespace clangd
} // namespace clang
