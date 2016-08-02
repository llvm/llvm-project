//===--- UnnecessaryValueParamCheck.cpp - clang-tidy-----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnnecessaryValueParamCheck.h"

#include "../utils/DeclRefExprUtils.h"
#include "../utils/FixItHintUtils.h"
#include "../utils/Matchers.h"
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

template <typename S>
bool isSubset(const S &SubsetCandidate, const S &SupersetCandidate) {
  for (const auto &E : SubsetCandidate)
    if (SupersetCandidate.count(E) == 0)
      return false;
  return true;
}

} // namespace

UnnecessaryValueParamCheck::UnnecessaryValueParamCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeStyle(utils::IncludeSorter::parseIncludeStyle(
          Options.get("IncludeStyle", "llvm"))) {}

void UnnecessaryValueParamCheck::registerMatchers(MatchFinder *Finder) {
  const auto ExpensiveValueParamDecl =
      parmVarDecl(hasType(hasCanonicalType(allOf(matchers::isExpensiveToCopy(),
                                                 unless(referenceType())))),
                  decl().bind("param"));
  Finder->addMatcher(
      functionDecl(isDefinition(), unless(cxxMethodDecl(isOverride())),
                   unless(isInstantiated()),
                   has(typeLoc(forEach(ExpensiveValueParamDecl))),
                   decl().bind("functionDecl")),
      this);
}

void UnnecessaryValueParamCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");
  const size_t Index = std::find(Function->parameters().begin(),
                                 Function->parameters().end(), Param) -
                       Function->parameters().begin();
  bool IsConstQualified =
      Param->getType().getCanonicalType().isConstQualified();

  // Skip declarations delayed by late template parsing without a body.
  if (!Function->getBody())
    return;

  // Do not trigger on non-const value parameters when:
  // 1. they are in a constructor definition since they can likely trigger
  //    misc-move-constructor-init which will suggest to move the argument.
  if (!IsConstQualified && (llvm::isa<CXXConstructorDecl>(Function) ||
                            !Function->doesThisDeclarationHaveABody()))
    return;

  auto AllDeclRefExprs = utils::decl_ref_expr::allDeclRefExprs(
      *Param, *Function->getBody(), *Result.Context);
  auto ConstDeclRefExprs = utils::decl_ref_expr::constReferenceDeclRefExprs(
      *Param, *Function->getBody(), *Result.Context);
  // 2. they are not only used as const.
  if (!isSubset(AllDeclRefExprs, ConstDeclRefExprs))
    return;

  // If the parameter is non-const, check if it has a move constructor and is
  // only referenced once to copy-construct another object or whether it has a
  // move assignment operator and is only referenced once when copy-assigned.
  // In this case wrap DeclRefExpr with std::move() to avoid the unnecessary
  // copy.
  if (!IsConstQualified) {
    auto CanonicalType = Param->getType().getCanonicalType();
    if (AllDeclRefExprs.size() == 1 &&
        ((utils::type_traits::hasNonTrivialMoveConstructor(CanonicalType) &&
          utils::decl_ref_expr::isCopyConstructorArgument(
              **AllDeclRefExprs.begin(), *Function->getBody(),
              *Result.Context)) ||
         (utils::type_traits::hasNonTrivialMoveAssignment(CanonicalType) &&
          utils::decl_ref_expr::isCopyAssignmentArgument(
              **AllDeclRefExprs.begin(), *Function->getBody(),
              *Result.Context)))) {
      handleMoveFix(*Param, **AllDeclRefExprs.begin(), *Result.Context);
      return;
    }
  }

  auto Diag =
      diag(Param->getLocation(),
           IsConstQualified ? "the const qualified parameter %0 is "
                              "copied for each invocation; consider "
                              "making it a reference"
                            : "the parameter %0 is copied for each "
                              "invocation but only used as a const reference; "
                              "consider making it a const reference")
      << paramNameOrIndex(Param->getName(), Index);
  // Do not propose fixes in macros since we cannot place them correctly, or if
  // function is virtual as it might break overrides.
  const auto *Method = llvm::dyn_cast<CXXMethodDecl>(Function);
  if (Param->getLocStart().isMacroID() || (Method && Method->isVirtual()))
    return;
  for (const auto *FunctionDecl = Function; FunctionDecl != nullptr;
       FunctionDecl = FunctionDecl->getPreviousDecl()) {
    const auto &CurrentParam = *FunctionDecl->getParamDecl(Index);
    Diag << utils::fixit::changeVarDeclToReference(CurrentParam,
                                                           *Result.Context);
    if (!IsConstQualified)
      Diag << utils::fixit::changeVarDeclToConst(CurrentParam);
  }
}

void UnnecessaryValueParamCheck::registerPPCallbacks(
    CompilerInstance &Compiler) {
  Inserter.reset(new utils::IncludeInserter(
      Compiler.getSourceManager(), Compiler.getLangOpts(), IncludeStyle));
  Compiler.getPreprocessor().addPPCallbacks(Inserter->CreatePPCallbacks());
}

void UnnecessaryValueParamCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle",
                utils::IncludeSorter::toString(IncludeStyle));
}

void UnnecessaryValueParamCheck::handleMoveFix(const ParmVarDecl &Var,
                                               const DeclRefExpr &CopyArgument,
                                               const ASTContext &Context) {
  auto Diag = diag(CopyArgument.getLocStart(),
                   "parameter %0 is passed by value and only copied once; "
                   "consider moving it to avoid unnecessary copies")
              << &Var;
  // Do not propose fixes in macros since we cannot place them correctly.
  if (CopyArgument.getLocStart().isMacroID())
    return;
  const auto &SM = Context.getSourceManager();
  auto EndLoc = Lexer::getLocForEndOfToken(CopyArgument.getLocation(), 0, SM,
                                           Context.getLangOpts());
  Diag << FixItHint::CreateInsertion(CopyArgument.getLocStart(), "std::move(")
       << FixItHint::CreateInsertion(EndLoc, ")");
  if (auto IncludeFixit = Inserter->CreateIncludeInsertion(
          SM.getFileID(CopyArgument.getLocStart()), "utility",
          /*IsAngled=*/true))
    Diag << *IncludeFixit;
}

} // namespace performance
} // namespace tidy
} // namespace clang
