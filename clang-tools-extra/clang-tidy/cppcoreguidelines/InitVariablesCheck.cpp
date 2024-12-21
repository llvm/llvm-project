//===--- InitVariablesCheck.cpp - clang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InitVariablesCheck.h"

#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

namespace {
AST_MATCHER(VarDecl, isLocalVarDecl) { return Node.isLocalVarDecl(); }
} // namespace

InitVariablesCheck::InitVariablesCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()),
      MathHeader(Options.get("MathHeader", "<math.h>")) {}

void InitVariablesCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
  Options.store(Opts, "MathHeader", MathHeader);
}

void InitVariablesCheck::registerMatchers(MatchFinder *Finder) {
  std::string BadDecl = "badDecl";
  Finder->addMatcher(
      varDecl(unless(hasInitializer(anything())), unless(isInstantiated()),
              isLocalVarDecl(), unless(isStaticLocal()), isDefinition(),
              unless(hasParent(cxxCatchStmt())),
              optionally(hasParent(declStmt(hasParent(
                  cxxForRangeStmt(hasLoopVariable(varDecl().bind(BadDecl))))))),
              unless(equalsBoundNode(BadDecl)))
          .bind("vardecl"),
      this);
}

void InitVariablesCheck::registerPPCallbacks(const SourceManager &SM,
                                             Preprocessor *PP,
                                             Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void InitVariablesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("vardecl");
  const ASTContext &Context = *Result.Context;
  const SourceManager &Source = Context.getSourceManager();

  // Clang diagnostic error may cause the variable to be an invalid int vardecl
  if (MatchedDecl->isInvalidDecl())
    return;

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
  std::optional<const char *> InitializationString;
  bool AddMathInclude = false;

  if (TypePtr->isEnumeralType())
    InitializationString = nullptr;
  else if (TypePtr->isBooleanType())
    InitializationString = " = false";
  else if (TypePtr->isIntegerType())
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
        << MatchedDecl;
    if (*InitializationString != nullptr)
      Diagnostic << FixItHint::CreateInsertion(
          utils::lexer::findNextTerminator(MatchedDecl->getLocation(),
                                           *Result.SourceManager,
                                           Result.Context->getLangOpts()),
          *InitializationString);
    if (AddMathInclude) {
      Diagnostic << IncludeInserter.createIncludeInsertion(
          Source.getFileID(MatchedDecl->getBeginLoc()), MathHeader);
    }
  }
}
} // namespace clang::tidy::cppcoreguidelines
