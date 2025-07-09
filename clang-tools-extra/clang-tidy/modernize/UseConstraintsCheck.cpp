//===--- UseConstraintsCheck.cpp - clang-tidy -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseConstraintsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

#include "../utils/LexerUtils.h"

#include <optional>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

struct EnableIfData {
  TemplateSpecializationTypeLoc Loc;
  TypeLoc Outer;
};

namespace {
AST_MATCHER(FunctionDecl, hasOtherDeclarations) {
  auto It = Node.redecls_begin();
  auto EndIt = Node.redecls_end();

  if (It == EndIt)
    return false;

  ++It;
  return It != EndIt;
}
} // namespace

void UseConstraintsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionTemplateDecl(
          // Skip external libraries included as system headers
          unless(isExpansionInSystemHeader()),
          has(functionDecl(unless(hasOtherDeclarations()), isDefinition(),
                           hasReturnTypeLoc(typeLoc().bind("return")))
                  .bind("function")))
          .bind("functionTemplate"),
      this);
}

static std::optional<TemplateSpecializationTypeLoc>
matchEnableIfSpecializationImplTypename(TypeLoc TheType) {
  if (const auto Dep = TheType.getAs<DependentNameTypeLoc>()) {
    const IdentifierInfo *Identifier = Dep.getTypePtr()->getIdentifier();
    ElaboratedTypeKeyword Keyword = Dep.getTypePtr()->getKeyword();
    if (!Identifier || Identifier->getName() != "type" ||
        (Keyword != ElaboratedTypeKeyword::Typename &&
         Keyword != ElaboratedTypeKeyword::None)) {
      return std::nullopt;
    }
    TheType = Dep.getQualifierLoc().getTypeLoc();
    if (TheType.isNull())
      return std::nullopt;
  }

  if (const auto SpecializationLoc =
          TheType.getAs<TemplateSpecializationTypeLoc>()) {

    const auto *Specialization =
        dyn_cast<TemplateSpecializationType>(SpecializationLoc.getTypePtr());
    if (!Specialization)
      return std::nullopt;

    const TemplateDecl *TD =
        Specialization->getTemplateName().getAsTemplateDecl();
    if (!TD || TD->getName() != "enable_if")
      return std::nullopt;

    int NumArgs = SpecializationLoc.getNumArgs();
    if (NumArgs != 1 && NumArgs != 2)
      return std::nullopt;

    return SpecializationLoc;
  }
  return std::nullopt;
}

static std::optional<TemplateSpecializationTypeLoc>
matchEnableIfSpecializationImplTrait(TypeLoc TheType) {
  if (const auto Elaborated = TheType.getAs<ElaboratedTypeLoc>())
    TheType = Elaborated.getNamedTypeLoc();

  if (const auto SpecializationLoc =
          TheType.getAs<TemplateSpecializationTypeLoc>()) {

    const auto *Specialization =
        dyn_cast<TemplateSpecializationType>(SpecializationLoc.getTypePtr());
    if (!Specialization)
      return std::nullopt;

    const TemplateDecl *TD =
        Specialization->getTemplateName().getAsTemplateDecl();
    if (!TD || TD->getName() != "enable_if_t")
      return std::nullopt;

    if (!Specialization->isTypeAlias())
      return std::nullopt;

    if (const auto *AliasedType =
            dyn_cast<DependentNameType>(Specialization->getAliasedType())) {
      ElaboratedTypeKeyword Keyword = AliasedType->getKeyword();
      if (AliasedType->getIdentifier()->getName() != "type" ||
          (Keyword != ElaboratedTypeKeyword::Typename &&
           Keyword != ElaboratedTypeKeyword::None)) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
    int NumArgs = SpecializationLoc.getNumArgs();
    if (NumArgs != 1 && NumArgs != 2)
      return std::nullopt;

    return SpecializationLoc;
  }
  return std::nullopt;
}

static std::optional<TemplateSpecializationTypeLoc>
matchEnableIfSpecializationImpl(TypeLoc TheType) {
  if (auto EnableIf = matchEnableIfSpecializationImplTypename(TheType))
    return EnableIf;
  return matchEnableIfSpecializationImplTrait(TheType);
}

static std::optional<EnableIfData>
matchEnableIfSpecialization(TypeLoc TheType) {
  if (const auto Pointer = TheType.getAs<PointerTypeLoc>())
    TheType = Pointer.getPointeeLoc();
  else if (const auto Reference = TheType.getAs<ReferenceTypeLoc>())
    TheType = Reference.getPointeeLoc();
  if (const auto Qualified = TheType.getAs<QualifiedTypeLoc>())
    TheType = Qualified.getUnqualifiedLoc();

  if (auto EnableIf = matchEnableIfSpecializationImpl(TheType))
    return EnableIfData{std::move(*EnableIf), TheType};
  return std::nullopt;
}

static std::pair<std::optional<EnableIfData>, const Decl *>
matchTrailingTemplateParam(const FunctionTemplateDecl *FunctionTemplate) {
  // For non-type trailing param, match very specifically
  // 'template <..., enable_if_type<Condition, Type> = Default>' where
  // enable_if_type is 'enable_if' or 'enable_if_t'. E.g., 'template <typename
  // T, enable_if_t<is_same_v<T, bool>, int*> = nullptr>
  //
  // Otherwise, match a trailing default type arg.
  // E.g., 'template <typename T, typename = enable_if_t<is_same_v<T, bool>>>'

  const TemplateParameterList *TemplateParams =
      FunctionTemplate->getTemplateParameters();
  if (TemplateParams->size() == 0)
    return {};

  const NamedDecl *LastParam =
      TemplateParams->getParam(TemplateParams->size() - 1);
  if (const auto *LastTemplateParam =
          dyn_cast<NonTypeTemplateParmDecl>(LastParam)) {

    if (!LastTemplateParam->hasDefaultArgument() ||
        !LastTemplateParam->getName().empty())
      return {};

    return {matchEnableIfSpecialization(
                LastTemplateParam->getTypeSourceInfo()->getTypeLoc()),
            LastTemplateParam};
  }
  if (const auto *LastTemplateParam =
          dyn_cast<TemplateTypeParmDecl>(LastParam)) {
    if (LastTemplateParam->hasDefaultArgument() &&
        LastTemplateParam->getIdentifier() == nullptr) {
      return {
          matchEnableIfSpecialization(LastTemplateParam->getDefaultArgument()
                                          .getTypeSourceInfo()
                                          ->getTypeLoc()),
          LastTemplateParam};
    }
  }
  return {};
}

template <typename T>
static SourceLocation getRAngleFileLoc(const SourceManager &SM,
                                       const T &Element) {
  // getFileLoc handles the case where the RAngle loc is part of a synthesized
  // '>>', which ends up allocating a 'scratch space' buffer in the source
  // manager.
  return SM.getFileLoc(Element.getRAngleLoc());
}

static SourceRange
getConditionRange(ASTContext &Context,
                  const TemplateSpecializationTypeLoc &EnableIf) {
  // TemplateArgumentLoc's SourceRange End is the location of the last token
  // (per UnqualifiedId docs). E.g., in `enable_if<AAA && BBB>`, the End
  // location will be the first 'B' in 'BBB'.
  const LangOptions &LangOpts = Context.getLangOpts();
  const SourceManager &SM = Context.getSourceManager();
  if (EnableIf.getNumArgs() > 1) {
    TemplateArgumentLoc NextArg = EnableIf.getArgLoc(1);
    return {EnableIf.getLAngleLoc().getLocWithOffset(1),
            utils::lexer::findPreviousTokenKind(
                NextArg.getSourceRange().getBegin(), SM, LangOpts, tok::comma)};
  }

  return {EnableIf.getLAngleLoc().getLocWithOffset(1),
          getRAngleFileLoc(SM, EnableIf)};
}

static SourceRange getTypeRange(ASTContext &Context,
                                const TemplateSpecializationTypeLoc &EnableIf) {
  TemplateArgumentLoc Arg = EnableIf.getArgLoc(1);
  const LangOptions &LangOpts = Context.getLangOpts();
  const SourceManager &SM = Context.getSourceManager();
  return {utils::lexer::findPreviousTokenKind(Arg.getSourceRange().getBegin(),
                                              SM, LangOpts, tok::comma)
              .getLocWithOffset(1),
          getRAngleFileLoc(SM, EnableIf)};
}

// Returns the original source text of the second argument of a call to
// enable_if_t. E.g., in enable_if_t<Condition, TheType>, this function
// returns 'TheType'.
static std::optional<StringRef>
getTypeText(ASTContext &Context,
            const TemplateSpecializationTypeLoc &EnableIf) {
  if (EnableIf.getNumArgs() > 1) {
    const LangOptions &LangOpts = Context.getLangOpts();
    const SourceManager &SM = Context.getSourceManager();
    bool Invalid = false;
    StringRef Text = Lexer::getSourceText(CharSourceRange::getCharRange(
                                              getTypeRange(Context, EnableIf)),
                                          SM, LangOpts, &Invalid)
                         .trim();
    if (Invalid)
      return std::nullopt;

    return Text;
  }

  return "void";
}

static std::optional<SourceLocation>
findInsertionForConstraint(const FunctionDecl *Function, ASTContext &Context) {
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  if (const auto *Constructor = dyn_cast<CXXConstructorDecl>(Function)) {
    for (const CXXCtorInitializer *Init : Constructor->inits()) {
      if (Init->getSourceOrder() == 0)
        return utils::lexer::findPreviousTokenKind(Init->getSourceLocation(),
                                                   SM, LangOpts, tok::colon);
    }
    if (!Constructor->inits().empty())
      return std::nullopt;
  }
  if (Function->isDeleted()) {
    SourceLocation FunctionEnd = Function->getSourceRange().getEnd();
    return utils::lexer::findNextAnyTokenKind(FunctionEnd, SM, LangOpts,
                                              tok::equal, tok::equal);
  }
  const Stmt *Body = Function->getBody();
  if (!Body)
    return std::nullopt;

  return Body->getBeginLoc();
}

static bool isPrimaryExpression(const Expr *Expression) {
  // This function is an incomplete approximation of checking whether
  // an Expr is a primary expression. In particular, if this function
  // returns true, the expression is a primary expression. The converse
  // is not necessarily true.

  if (const auto *Cast = dyn_cast<ImplicitCastExpr>(Expression))
    Expression = Cast->getSubExprAsWritten();
  if (isa<ParenExpr, DependentScopeDeclRefExpr>(Expression))
    return true;

  return false;
}

// Return the original source text of an enable_if_t condition, i.e., the
// first template argument). For example, in
// 'enable_if_t<FirstCondition || SecondCondition, AType>', the text
// the text 'FirstCondition || SecondCondition' is returned.
static std::optional<std::string> getConditionText(const Expr *ConditionExpr,
                                                   SourceRange ConditionRange,
                                                   ASTContext &Context) {
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  SourceLocation PrevTokenLoc = ConditionRange.getEnd();
  if (PrevTokenLoc.isInvalid())
    return std::nullopt;

  const bool SkipComments = false;
  Token PrevToken;
  std::tie(PrevToken, PrevTokenLoc) = utils::lexer::getPreviousTokenAndStart(
      PrevTokenLoc, SM, LangOpts, SkipComments);
  bool EndsWithDoubleSlash =
      PrevToken.is(tok::comment) &&
      Lexer::getSourceText(CharSourceRange::getCharRange(
                               PrevTokenLoc, PrevTokenLoc.getLocWithOffset(2)),
                           SM, LangOpts) == "//";

  bool Invalid = false;
  llvm::StringRef ConditionText = Lexer::getSourceText(
      CharSourceRange::getCharRange(ConditionRange), SM, LangOpts, &Invalid);
  if (Invalid)
    return std::nullopt;

  auto AddParens = [&](llvm::StringRef Text) -> std::string {
    if (isPrimaryExpression(ConditionExpr))
      return Text.str();
    return "(" + Text.str() + ")";
  };

  if (EndsWithDoubleSlash)
    return AddParens(ConditionText);
  return AddParens(ConditionText.trim());
}

// Handle functions that return enable_if_t, e.g.,
//   template <...>
//   enable_if_t<Condition, ReturnType> function();
//
// Return a vector of FixItHints if the code can be replaced with
// a C++20 requires clause. In the example above, returns FixItHints
// to result in
//   template <...>
//   ReturnType function() requires Condition {}
static std::vector<FixItHint> handleReturnType(const FunctionDecl *Function,
                                               const TypeLoc &ReturnType,
                                               const EnableIfData &EnableIf,
                                               ASTContext &Context) {
  TemplateArgumentLoc EnableCondition = EnableIf.Loc.getArgLoc(0);

  SourceRange ConditionRange = getConditionRange(Context, EnableIf.Loc);

  std::optional<std::string> ConditionText = getConditionText(
      EnableCondition.getSourceExpression(), ConditionRange, Context);
  if (!ConditionText)
    return {};

  std::optional<StringRef> TypeText = getTypeText(Context, EnableIf.Loc);
  if (!TypeText)
    return {};

  SmallVector<AssociatedConstraint, 3> ExistingConstraints;
  Function->getAssociatedConstraints(ExistingConstraints);
  if (!ExistingConstraints.empty()) {
    // FIXME - Support adding new constraints to existing ones. Do we need to
    // consider subsumption?
    return {};
  }

  std::optional<SourceLocation> ConstraintInsertionLoc =
      findInsertionForConstraint(Function, Context);
  if (!ConstraintInsertionLoc)
    return {};

  std::vector<FixItHint> FixIts;
  FixIts.push_back(FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(EnableIf.Outer.getSourceRange()),
      *TypeText));
  FixIts.push_back(FixItHint::CreateInsertion(
      *ConstraintInsertionLoc, "requires " + *ConditionText + " "));
  return FixIts;
}

// Handle enable_if_t in a trailing template parameter, e.g.,
//   template <..., enable_if_t<Condition, Type> = Type{}>
//   ReturnType function();
//
// Return a vector of FixItHints if the code can be replaced with
// a C++20 requires clause. In the example above, returns FixItHints
// to result in
//   template <...>
//   ReturnType function() requires Condition {}
static std::vector<FixItHint>
handleTrailingTemplateType(const FunctionTemplateDecl *FunctionTemplate,
                           const FunctionDecl *Function,
                           const Decl *LastTemplateParam,
                           const EnableIfData &EnableIf, ASTContext &Context) {
  SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  TemplateArgumentLoc EnableCondition = EnableIf.Loc.getArgLoc(0);

  SourceRange ConditionRange = getConditionRange(Context, EnableIf.Loc);

  std::optional<std::string> ConditionText = getConditionText(
      EnableCondition.getSourceExpression(), ConditionRange, Context);
  if (!ConditionText)
    return {};

  SmallVector<AssociatedConstraint, 3> ExistingConstraints;
  Function->getAssociatedConstraints(ExistingConstraints);
  if (!ExistingConstraints.empty()) {
    // FIXME - Support adding new constraints to existing ones. Do we need to
    // consider subsumption?
    return {};
  }

  SourceRange RemovalRange;
  const TemplateParameterList *TemplateParams =
      FunctionTemplate->getTemplateParameters();
  if (!TemplateParams || TemplateParams->size() == 0)
    return {};

  if (TemplateParams->size() == 1) {
    RemovalRange =
        SourceRange(TemplateParams->getTemplateLoc(),
                    getRAngleFileLoc(SM, *TemplateParams).getLocWithOffset(1));
  } else {
    RemovalRange =
        SourceRange(utils::lexer::findPreviousTokenKind(
                        LastTemplateParam->getSourceRange().getBegin(), SM,
                        LangOpts, tok::comma),
                    getRAngleFileLoc(SM, *TemplateParams));
  }

  std::optional<SourceLocation> ConstraintInsertionLoc =
      findInsertionForConstraint(Function, Context);
  if (!ConstraintInsertionLoc)
    return {};

  std::vector<FixItHint> FixIts;
  FixIts.push_back(
      FixItHint::CreateRemoval(CharSourceRange::getCharRange(RemovalRange)));
  FixIts.push_back(FixItHint::CreateInsertion(
      *ConstraintInsertionLoc, "requires " + *ConditionText + " "));
  return FixIts;
}

void UseConstraintsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FunctionTemplate =
      Result.Nodes.getNodeAs<FunctionTemplateDecl>("functionTemplate");
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("function");
  const auto *ReturnType = Result.Nodes.getNodeAs<TypeLoc>("return");
  if (!FunctionTemplate || !Function || !ReturnType)
    return;

  // Check for
  //
  //   Case 1. Return type of function
  //
  //     template <...>
  //     enable_if_t<Condition, ReturnType>::type function() {}
  //
  //   Case 2. Trailing template parameter
  //
  //     template <..., enable_if_t<Condition, Type> = Type{}>
  //     ReturnType function() {}
  //
  //     or
  //
  //     template <..., typename = enable_if_t<Condition, void>>
  //     ReturnType function() {}
  //

  // Case 1. Return type of function
  if (auto EnableIf = matchEnableIfSpecialization(*ReturnType)) {
    diag(ReturnType->getBeginLoc(),
         "use C++20 requires constraints instead of enable_if")
        << handleReturnType(Function, *ReturnType, *EnableIf, *Result.Context);
    return;
  }

  // Case 2. Trailing template parameter
  if (auto [EnableIf, LastTemplateParam] =
          matchTrailingTemplateParam(FunctionTemplate);
      EnableIf && LastTemplateParam) {
    diag(LastTemplateParam->getSourceRange().getBegin(),
         "use C++20 requires constraints instead of enable_if")
        << handleTrailingTemplateType(FunctionTemplate, Function,
                                      LastTemplateParam, *EnableIf,
                                      *Result.Context);
    return;
  }
}

} // namespace clang::tidy::modernize
