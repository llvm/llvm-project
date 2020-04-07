//===--- InitVariablesCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InitVariablesCheck.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

namespace {
AST_MATCHER(VarDecl, isLocalVarDecl) { return Node.isLocalVarDecl(); }
} // namespace

InitVariablesCheck::InitVariablesCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeStyle(Options.getLocalOrGlobal("IncludeStyle",
                                            utils::IncludeSorter::getMapping(),
                                            utils::IncludeSorter::IS_LLVM)),
      MathHeader(Options.get("MathHeader", "math.h")) {}

void InitVariablesCheck::registerMatchers(MatchFinder *Finder) {
  std::string BadDecl = "badDecl";
  Finder->addMatcher(
      varDecl(unless(hasInitializer(anything())), unless(isInstantiated()),
              isLocalVarDecl(), unless(isStaticLocal()), isDefinition(),
              optionally(hasParent(declStmt(hasParent(
                  cxxForRangeStmt(hasLoopVariable(varDecl().bind(BadDecl))))))),
              unless(equalsBoundNode(BadDecl)))
          .bind("vardecl"),
      this);
}

void InitVariablesCheck::registerPPCallbacks(const SourceManager &SM,
                                             Preprocessor *PP,
                                             Preprocessor *ModuleExpanderPP) {
  IncludeInserter =
      std::make_unique<utils::IncludeInserter>(SM, getLangOpts(), IncludeStyle);
  PP->addPPCallbacks(IncludeInserter->CreatePPCallbacks());
}

void InitVariablesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("vardecl");
  const ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();

  // We want to warn about cases where the type name
  // comes from a macro like this:
  //
  // TYPENAME_FROM_MACRO var;
  //
  // but not if the entire declaration comes from
  // one:
  //
  // DEFINE_SOME_VARIABLE();
  //
  // or if the definition comes from a macro like SWAP
  // that uses an internal temporary variable.
  //
  // Thus check that the variable name does
  // not come from a macro expansion.
  if (MatchedDecl->getEndLoc().isMacroID())
    return;

  QualType TypePtr = MatchedDecl->getType();
  const char *InitializationString = nullptr;
  bool AddMathInclude = false;

  if (TypePtr->isIntegerType())
    InitializationString = " = 0";
  else if (TypePtr->isFloatingType()) {
    InitializationString = " = NAN";
    AddMathInclude = true;
  } else if (TypePtr->isAnyPointerType()) {
    if (getLangOpts().CPlusPlus11)
      InitializationString = " = nullptr";
    else
      InitializationString = " = NULL";
  }

  if (InitializationString) {
    auto Diagnostic =
        diag(MatchedDecl->getLocation(), "variable %0 is not initialized")
        << MatchedDecl
        << FixItHint::CreateInsertion(
               MatchedDecl->getLocation().getLocWithOffset(
                   MatchedDecl->getName().size()),
               InitializationString);
    if (AddMathInclude) {
      Diagnostic << IncludeInserter->CreateIncludeInsertion(
          Source.getFileID(MatchedDecl->getBeginLoc()), MathHeader, false);
    }
  }
}

} // namespace cppcoreguidelines
} // namespace tidy
} // namespace clang
