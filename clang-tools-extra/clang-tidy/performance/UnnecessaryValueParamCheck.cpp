//===--- UnnecessaryValueParamCheck.cpp - clang-tidy-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnnecessaryValueParamCheck.h"

#include "../utils/DeclRefExprUtils.h"
#include "../utils/FixItHintUtils.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "../utils/TypeTraits.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

namespace {

std::string paramNameOrIndex(StringRef Name, size_t Index) {
  return (Name.empty() ? llvm::Twine('#') + llvm::Twine(Index + 1)
                       : llvm::Twine('\'') + Name + llvm::Twine('\''))
      .str();
}

bool isReferencedOutsideOfCallExpr(const FunctionDecl &Function,
                                   ASTContext &Context) {
  auto Matches = match(declRefExpr(to(functionDecl(equalsNode(&Function))),
                                   unless(hasAncestor(callExpr()))),
                       Context);
  return !Matches.empty();
}

bool hasLoopStmtAncestor(const DeclRefExpr &DeclRef, const Decl &Decl,
                         ASTContext &Context) {
  auto Matches = match(
      traverse(ast_type_traits::TK_AsIs,
               decl(forEachDescendant(declRefExpr(
                   equalsNode(&DeclRef),
                   unless(hasAncestor(stmt(anyOf(forStmt(), cxxForRangeStmt(),
                                                 whileStmt(), doStmt())))))))),
      Decl, Context);
  return Matches.empty();
}

bool isExplicitTemplateSpecialization(const FunctionDecl &Function) {
  if (const auto *SpecializationInfo = Function.getTemplateSpecializationInfo())
    if (SpecializationInfo->getTemplateSpecializationKind() ==
        TSK_ExplicitSpecialization)
      return true;
  if (const auto *Method = llvm::dyn_cast<CXXMethodDecl>(&Function))
    if (Method->getTemplatedKind() == FunctionDecl::TK_MemberSpecialization &&
        Method->getMemberSpecializationInfo()->isExplicitSpecialization())
      return true;
  return false;
}

} // namespace

UnnecessaryValueParamCheck::UnnecessaryValueParamCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeStyle(Options.getLocalOrGlobal("IncludeStyle",
                                            utils::IncludeSorter::getMapping(),
                                            utils::IncludeSorter::IS_LLVM)),
      AllowedTypes(
          utils::options::parseStringList(Options.get("AllowedTypes", ""))) {}

void UnnecessaryValueParamCheck::registerMatchers(MatchFinder *Finder) {
  const auto ExpensiveValueParamDecl = parmVarDecl(
      hasType(qualType(
          hasCanonicalType(matchers::isExpensiveToCopy()),
          unless(anyOf(hasCanonicalType(referenceType()),
                       hasDeclaration(namedDecl(
                           matchers::matchesAnyListedName(AllowedTypes))))))),
      decl().bind("param"));
  Finder->addMatcher(
      traverse(
          ast_type_traits::TK_AsIs,
          functionDecl(hasBody(stmt()), isDefinition(), unless(isImplicit()),
                       unless(cxxMethodDecl(anyOf(isOverride(), isFinal()))),
                       has(typeLoc(forEach(ExpensiveValueParamDecl))),
                       unless(isInstantiated()), decl().bind("functionDecl"))),
      this);
}

void UnnecessaryValueParamCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");

  TraversalKindScope RAII(*Result.Context, ast_type_traits::TK_AsIs);

  FunctionParmMutationAnalyzer &Analyzer =
      MutationAnalyzers.try_emplace(Function, *Function, *Result.Context)
          .first->second;
  if (Analyzer.isMutated(Param))
    return;

  const bool IsConstQualified =
      Param->getType().getCanonicalType().isConstQualified();

  // If the parameter is non-const, check if it has a move constructor and is
  // only referenced once to copy-construct another object or whether it has a
  // move assignment operator and is only referenced once when copy-assigned.
  // In this case wrap DeclRefExpr with std::move() to avoid the unnecessary
  // copy.
  if (!IsConstQualified) {
    auto AllDeclRefExprs = utils::decl_ref_expr::allDeclRefExprs(
        *Param, *Function, *Result.Context);
    if (AllDeclRefExprs.size() == 1) {
      auto CanonicalType = Param->getType().getCanonicalType();
      const auto &DeclRefExpr = **AllDeclRefExprs.begin();

      if (!hasLoopStmtAncestor(DeclRefExpr, *Function, *Result.Context) &&
          ((utils::type_traits::hasNonTrivialMoveConstructor(CanonicalType) &&
            utils::decl_ref_expr::isCopyConstructorArgument(
                DeclRefExpr, *Function, *Result.Context)) ||
           (utils::type_traits::hasNonTrivialMoveAssignment(CanonicalType) &&
            utils::decl_ref_expr::isCopyAssignmentArgument(
                DeclRefExpr, *Function, *Result.Context)))) {
        handleMoveFix(*Param, DeclRefExpr, *Result.Context);
        return;
      }
    }
  }

  const size_t Index = std::find(Function->parameters().begin(),
                                 Function->parameters().end(), Param) -
                       Function->parameters().begin();

  auto Diag =
      diag(Param->getLocation(),
           IsConstQualified ? "the const qualified parameter %0 is "
                              "copied for each invocation; consider "
                              "making it a reference"
                            : "the parameter %0 is copied for each "
                              "invocation but only used as a const reference; "
                              "consider making it a const reference")
      << paramNameOrIndex(Param->getName(), Index);
  // Do not propose fixes when:
  // 1. the ParmVarDecl is in a macro, since we cannot place them correctly
  // 2. the function is virtual as it might break overrides
  // 3. the function is referenced outside of a call expression within the
  //    compilation unit as the signature change could introduce build errors.
  // 4. the function is an explicit template specialization.
  const auto *Method = llvm::dyn_cast<CXXMethodDecl>(Function);
  if (Param->getBeginLoc().isMacroID() || (Method && Method->isVirtual()) ||
      isReferencedOutsideOfCallExpr(*Function, *Result.Context) ||
      isExplicitTemplateSpecialization(*Function))
    return;
  for (const auto *FunctionDecl = Function; FunctionDecl != nullptr;
       FunctionDecl = FunctionDecl->getPreviousDecl()) {
    const auto &CurrentParam = *FunctionDecl->getParamDecl(Index);
    Diag << utils::fixit::changeVarDeclToReference(CurrentParam,
                                                   *Result.Context);
    // The parameter of each declaration needs to be checked individually as to
    // whether it is const or not as constness can differ between definition and
    // declaration.
    if (!CurrentParam.getType().getCanonicalType().isConstQualified()) {
      if (llvm::Optional<FixItHint> Fix = utils::fixit::addQualifierToVarDecl(
              CurrentParam, *Result.Context, DeclSpec::TQ::TQ_const))
        Diag << *Fix;
    }
  }
}

void UnnecessaryValueParamCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter = std::make_unique<utils::IncludeInserter>(SM, getLangOpts(),
                                                      IncludeStyle);
  PP->addPPCallbacks(Inserter->CreatePPCallbacks());
}

void UnnecessaryValueParamCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeStyle,
                utils::IncludeSorter::getMapping());
  Options.store(Opts, "AllowedTypes",
                utils::options::serializeStringList(AllowedTypes));
}

void UnnecessaryValueParamCheck::onEndOfTranslationUnit() {
  MutationAnalyzers.clear();
}

void UnnecessaryValueParamCheck::handleMoveFix(const ParmVarDecl &Var,
                                               const DeclRefExpr &CopyArgument,
                                               const ASTContext &Context) {
  auto Diag = diag(CopyArgument.getBeginLoc(),
                   "parameter %0 is passed by value and only copied once; "
                   "consider moving it to avoid unnecessary copies")
              << &Var;
  // Do not propose fixes in macros since we cannot place them correctly.
  if (CopyArgument.getBeginLoc().isMacroID())
    return;
  const auto &SM = Context.getSourceManager();
  auto EndLoc = Lexer::getLocForEndOfToken(CopyArgument.getLocation(), 0, SM,
                                           Context.getLangOpts());
  Diag << FixItHint::CreateInsertion(CopyArgument.getBeginLoc(), "std::move(")
       << FixItHint::CreateInsertion(EndLoc, ")")
       << Inserter->CreateIncludeInsertion(
              SM.getFileID(CopyArgument.getBeginLoc()), "utility",
              /*IsAngled=*/true);
}

} // namespace performance
} // namespace tidy
} // namespace clang
