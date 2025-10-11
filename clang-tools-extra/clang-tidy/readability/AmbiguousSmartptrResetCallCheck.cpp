//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AmbiguousSmartptrResetCallCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_MATCHER(CXXMethodDecl, hasOnlyDefaultParameters) {
  for (const auto *Param : Node.parameters()) {
    if (!Param->hasDefaultArg())
      return false;
  }

  return true;
}

const auto DefaultSmartPointers = "::std::shared_ptr;::std::unique_ptr;"
                                  "::boost::shared_ptr";
} // namespace

AmbiguousSmartptrResetCallCheck::AmbiguousSmartptrResetCallCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SmartPointers(utils::options::parseStringList(
          Options.get("SmartPointers", DefaultSmartPointers))) {}

void AmbiguousSmartptrResetCallCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SmartPointers",
                utils::options::serializeStringList(SmartPointers));
}

void AmbiguousSmartptrResetCallCheck::registerMatchers(MatchFinder *Finder) {
  const auto IsSmartptr = hasAnyName(SmartPointers);

  const auto ResetMethod =
      cxxMethodDecl(hasName("reset"), hasOnlyDefaultParameters());

  const auto TypeWithReset =
      anyOf(cxxRecordDecl(
                anyOf(hasMethod(ResetMethod),
                      isDerivedFrom(cxxRecordDecl(hasMethod(ResetMethod))))),
            classTemplateSpecializationDecl(
                hasSpecializedTemplate(classTemplateDecl(has(ResetMethod)))));

  const auto SmartptrWithReset = expr(hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(classTemplateSpecializationDecl(
          IsSmartptr,
          hasTemplateArgument(
              0, templateArgument(refersToType(hasUnqualifiedDesugaredType(
                     recordType(hasDeclaration(TypeWithReset))))))))))));

  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(ResetMethod),
          unless(hasAnyArgument(expr(unless(cxxDefaultArgExpr())))),
          anyOf(on(cxxOperatorCallExpr(hasOverloadedOperatorName("->"),
                                       hasArgument(0, SmartptrWithReset))
                       .bind("ArrowOp")),
                on(SmartptrWithReset)))
          .bind("MemberCall"),
      this);
}

void AmbiguousSmartptrResetCallCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("MemberCall");
  assert(MemberCall);

  if (const auto *Arrow =
          Result.Nodes.getNodeAs<CXXOperatorCallExpr>("ArrowOp")) {
    const CharSourceRange SmartptrSourceRange =
        Lexer::getAsCharRange(Arrow->getArg(0)->getSourceRange(),
                              *Result.SourceManager, getLangOpts());

    diag(MemberCall->getBeginLoc(),
         "ambiguous call to 'reset()' on a pointee of a smart pointer, prefer "
         "more explicit approach");

    diag(MemberCall->getBeginLoc(),
         "consider dereferencing smart pointer to call 'reset' method "
         "of the pointee here",
         DiagnosticIDs::Note)
        << FixItHint::CreateInsertion(SmartptrSourceRange.getBegin(), "(*")
        << FixItHint::CreateInsertion(SmartptrSourceRange.getEnd(), ")")
        << FixItHint::CreateReplacement(
               CharSourceRange::getCharRange(
                   Arrow->getOperatorLoc(),
                   Arrow->getOperatorLoc().getLocWithOffset(2)),
               ".");
  } else {
    const auto *Member = cast<MemberExpr>(MemberCall->getCallee());
    assert(Member);

    diag(MemberCall->getBeginLoc(),
         "ambiguous call to 'reset()' on a smart pointer with pointee that "
         "also has a 'reset()' method, prefer more explicit approach");

    diag(MemberCall->getBeginLoc(),
         "consider assigning the pointer to 'nullptr' here",
         DiagnosticIDs::Note)
        << FixItHint::CreateReplacement(
               SourceRange(Member->getOperatorLoc(), Member->getOperatorLoc()),
               " =")
        << FixItHint::CreateReplacement(
               SourceRange(Member->getMemberLoc(), MemberCall->getEndLoc()),
               " nullptr");
  }
}

} // namespace clang::tidy::readability
