//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseFromRangeContainerConstructorCheck.h"

#include <optional>
#include <string>

#include "../ClangTidyCheck.h"
#include "../ClangTidyDiagnosticConsumer.h"
#include "../ClangTidyOptions.h"
#include "../utils/ASTUtils.h"
#include "../utils/IncludeSorter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"

namespace clang::tidy::modernize {

namespace {

using ast_matchers::argumentCountAtLeast;
using ast_matchers::cxxConstructExpr;
using ast_matchers::cxxConstructorDecl;
using ast_matchers::cxxRecordDecl;
using ast_matchers::hasAnyName;
using ast_matchers::hasDeclaration;
using ast_matchers::ofClass;

struct RangeObjectInfo {
  const Expr *Object;
  bool IsArrow;
  StringRef Name;
};

} // namespace

static std::optional<RangeObjectInfo> getRangeAndFunctionName(const Expr *E) {
  E = E->IgnoreParenImpCasts();
  const Expr *Base = nullptr;
  bool IsArrow = false;
  StringRef Name;
  if (const auto *MemberCall = dyn_cast<CXXMemberCallExpr>(E)) {
    if (const auto *ME = dyn_cast<MemberExpr>(
            MemberCall->getCallee()->IgnoreParenImpCasts())) {
      Base = ME->getBase()->IgnoreParenImpCasts();
      IsArrow = ME->isArrow();
      Name = ME->getMemberDecl()->getName();
    }
  } else if (const auto *Call = dyn_cast<CallExpr>(E)) {
    if (Call->getNumArgs() == 1 && Call->getDirectCallee()) {
      Base = Call->getArg(0)->IgnoreParenImpCasts();
      IsArrow = false;
      Name = Call->getDirectCallee()->getName();
    }
  }

  if (!Base)
    return std::nullopt;

  // PEEL LAYER: Handle Smart Pointers (overloaded operator->)
  // If the base is an operator call, we want the text of the underlying
  // pointer.
  if (const auto *OpCall = dyn_cast<CXXOperatorCallExpr>(Base)) {
    if (OpCall->getOperator() == OO_Arrow) {
      Base = OpCall->getArg(0)->IgnoreParenImpCasts();
      IsArrow = true;
    }
  }

  return RangeObjectInfo{Base, IsArrow, Name};
}

static QualType getValueType(QualType T) {
  if (const auto *Spec = T->getAs<TemplateSpecializationType>()) {
    const StringRef Name =
        Spec->getTemplateName().getAsTemplateDecl()->getName();
    if (Name == "map" || Name == "unordered_map")
      return {};

    if (Name == "unique_ptr") {
      if (!Spec->template_arguments().empty() &&
          Spec->template_arguments()[0].getKind() == TemplateArgument::Type)
        return getValueType(Spec->template_arguments()[0].getAsType());
      return {};
    }

    const ArrayRef<TemplateArgument> &Args = Spec->template_arguments();
    if (!Args.empty() && Args[0].getKind() == TemplateArgument::Type)
      return Args[0].getAsType();
  }
  return {};
}

UseFromRangeContainerConstructorCheck::UseFromRangeContainerConstructorCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               /*SelfContainedDiags=*/false) {}

void UseFromRangeContainerConstructorCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseFromRangeContainerConstructorCheck::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  auto ContainerNames =
      hasAnyName("::std::vector", "::std::deque", "::std::forward_list",
                 "::std::list", "::std::set", "::std::map",
                 "::std::unordered_set", "::std::unordered_map",
                 "::std::priority_queue", "::std::queue", "::std::stack",
                 "::std::basic_string", "::std::flat_set", "::std::flat_map");
  Finder->addMatcher(cxxConstructExpr(argumentCountAtLeast(2),
                                      hasDeclaration(cxxConstructorDecl(ofClass(
                                          cxxRecordDecl(ContainerNames)))))
                         .bind("ctor"),
                     this);
}

void UseFromRangeContainerConstructorCheck::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *CtorExpr = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
  std::optional<RangeObjectInfo> BeginInfo =
      getRangeAndFunctionName(CtorExpr->getArg(0));
  std::optional<RangeObjectInfo> EndInfo =
      getRangeAndFunctionName(CtorExpr->getArg(1));
  if (!BeginInfo || !EndInfo)
    return;

  if (!((BeginInfo->Name == "begin" && EndInfo->Name == "end") ||
        (BeginInfo->Name == "cbegin" && EndInfo->Name == "cend"))) {
    return;
  }

  if (!utils::areStatementsIdentical(BeginInfo->Object, EndInfo->Object,
                                     *Result.Context)) {
    return;
  }

  // Type compatibility check.
  //
  // 1) Same type, std::from_range works, warn.
  //
  //   std::set<std::string> source;
  //   std::vector<std::string> dest(source.begin(), source.end());
  //
  // 2) Needs explicit conversion, std::from_range doesn't work, so don't warn.
  //
  //   std::set<std::string_view> source;
  //   std::vector<std::string> dest(source.begin(), source.end());
  //
  // 3) Implicitly convertible, std::from_range works, but do not warn, since
  //   checking this case is hard in clang-tidy.
  //
  //   std::set<std::string> source;
  //   std::vector<std::string_view> dest(source.begin(), source.end());
  QualType SourceRangeType = BeginInfo->Object->getType();
  if (const auto *Type = SourceRangeType->getAs<PointerType>())
    SourceRangeType = Type->getPointeeType();
  const QualType SourceValueType = getValueType(SourceRangeType);

  if (const auto *DestSpec =
          CtorExpr->getType()->getAs<TemplateSpecializationType>()) {
    const StringRef Name =
        DestSpec->getTemplateName().getAsTemplateDecl()->getName();
    if ((Name == "map" || Name == "unordered_map") &&
        !SourceValueType.isNull()) {
      if (const auto *SourcePairSpec =
              SourceValueType->getAs<TemplateSpecializationType>()) {
        if (SourcePairSpec->getTemplateName().getAsTemplateDecl()->getName() ==
            "pair") {
          const QualType DestKeyType =
              DestSpec->template_arguments()[0].getAsType();
          const QualType SourceKeyType =
              SourcePairSpec->template_arguments()[0].getAsType();
          if (!ASTContext::hasSameUnqualifiedType(DestKeyType, SourceKeyType))
            return;
        }
      }
    }
  }

  const QualType DestValueType = getValueType(CtorExpr->getType());
  if (!DestValueType.isNull() && !SourceValueType.isNull() &&
      !ASTContext::hasSameUnqualifiedType(DestValueType, SourceValueType)) {
    return;
  }

  std::string BaseText =
      tooling::fixit::getText(*BeginInfo->Object, *Result.Context).str();
  if (BaseText.empty())
    return;

  StringRef BaseRef(BaseText);
  BaseRef.consume_back("->");
  BaseText = BaseRef.str();
  std::string Replacement = "std::from_range, ";
  if (BeginInfo->IsArrow) {
    // Determine if we need safety parentheses: *(p + 1) vs *p
    const bool SimpleIdentifier =
        BaseText.find_first_of(" +-*/%&|^") == std::string::npos;
    Replacement += SimpleIdentifier ? "*" + BaseText : "*(" + BaseText + ")";
  } else {
    Replacement += BaseText;
  }

  const DiagnosticBuilder Diag =
      diag(CtorExpr->getBeginLoc(),
           "use std::from_range for container construction");
  const SourceRange ArgRange(CtorExpr->getArg(0)->getBeginLoc(),
                             CtorExpr->getArg(1)->getEndLoc());
  Diag << FixItHint::CreateReplacement(ArgRange, Replacement);
  Diag << Inserter.createMainFileIncludeInsertion("<ranges>");
}

void UseFromRangeContainerConstructorCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

} // namespace clang::tidy::modernize
