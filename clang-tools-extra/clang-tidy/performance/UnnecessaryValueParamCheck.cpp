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
#include "../utils/ExprMutationAnalyzer.h"
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

bool isReferencedOutsideOfCallExpr(const FunctionDecl &Function,
                                   ASTContext &Context) {
  auto Matches = match(declRefExpr(to(functionDecl(equalsNode(&Function))),
                                   unless(hasAncestor(callExpr()))),
                       Context);
  return !Matches.empty();
}

bool hasLoopStmtAncestor(const DeclRefExpr &DeclRef, const Decl &Decl,
                         ASTContext &Context) {
  auto Matches =
      match(decl(forEachDescendant(declRefExpr(
                equalsNode(&DeclRef),
                unless(hasAncestor(stmt(anyOf(forStmt(), cxxForRangeStmt(),
                                              whileStmt(), doStmt()))))))),
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
      IncludeStyle(utils::IncludeSorter::parseIncludeStyle(
          Options.getLocalOrGlobal("IncludeStyle", "llvm"))) {}

void UnnecessaryValueParamCheck::registerMatchers(MatchFinder *Finder) {
  // This check is specific to C++ and doesn't apply to languages like
  // Objective-C.
  if (!getLangOpts().CPlusPlus)
    return;
  const auto ExpensiveValueParamDecl =
      parmVarDecl(hasType(hasCanonicalType(allOf(
                      unless(referenceType()), matchers::isExpensiveToCopy()))),
                  decl().bind("param"));
  Finder->addMatcher(
      functionDecl(hasBody(stmt()), isDefinition(), unless(isImplicit()),
                   unless(cxxMethodDecl(anyOf(isOverride(), isFinal()))),
                   has(typeLoc(forEach(ExpensiveValueParamDecl))),
                   unless(isInstantiated()), decl().bind("functionDecl")),
      this);
}

void UnnecessaryValueParamCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Param = Result.Nodes.getNodeAs<ParmVarDecl>("param");
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("functionDecl");

  // Do not trigger on non-const value parameters when they are mutated either
  // within the function body or within init expression(s) when the function is
  // a ctor.
  if (utils::ExprMutationAnalyzer(Function->getBody(), Result.Context)
          .isMutated(Param))
    return;
  // CXXCtorInitializer might also mutate Param but they're not part of function
  // body, so check them separately here.
  if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Function)) {
    for (const auto *Init : Ctor->inits()) {
      if (utils::ExprMutationAnalyzer(Init->getInit(), Result.Context)
              .isMutated(Param))
        return;
    }
  }

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
    if (!CurrentParam.getType().getCanonicalType().isConstQualified())
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
       << FixItHint::CreateInsertion(EndLoc, ")");
  if (auto IncludeFixit = Inserter->CreateIncludeInsertion(
          SM.getFileID(CopyArgument.getBeginLoc()), "utility",
          /*IsAngled=*/true))
    Diag << *IncludeFixit;
}

} // namespace performance
} // namespace tidy
} // namespace clang
