//===--- MakeSmartPtrCheck.cpp - clang-tidy--------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MakeSharedCheck.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

const char MakeSmartPtrCheck::PointerType[] = "pointerType";
const char MakeSmartPtrCheck::ConstructorCall[] = "constructorCall";
const char MakeSmartPtrCheck::NewExpression[] = "newExpression";

MakeSmartPtrCheck::MakeSmartPtrCheck(StringRef Name, ClangTidyContext *Context,
                                     std::string makeSmartPtrFunctionName)
    : ClangTidyCheck(Name, Context),
      makeSmartPtrFunctionName(std::move(makeSmartPtrFunctionName)) {}

void MakeSmartPtrCheck::registerMatchers(ast_matchers::MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus11)
    return;

  // Calling make_smart_ptr from within a member function of a type with a
  // private or protected constructor would be ill-formed.
  auto CanCallCtor = unless(has(ignoringImpCasts(cxxConstructExpr(
      hasDeclaration(decl(unless(isPublic())))))));

  Finder->addMatcher(
      cxxBindTemporaryExpr(has(ignoringParenImpCasts(
          cxxConstructExpr(
              hasType(getSmartPointerTypeMatcher()), argumentCountIs(1),
              hasArgument(0,
                          cxxNewExpr(hasType(pointsTo(qualType(hasCanonicalType(
                                         equalsBoundNode(PointerType))))),
                                     CanCallCtor)
                              .bind(NewExpression)))
              .bind(ConstructorCall)))),
      this);
}

void MakeSmartPtrCheck::check(const MatchFinder::MatchResult &Result) {
  // 'smart_ptr' refers to 'std::shared_ptr' or 'std::unique_ptr' or other
  // pointer, 'make_smart_ptr' refers to 'std::make_shared' or
  // 'std::make_unique' or other function that creates smart_ptr.

  SourceManager &SM = *Result.SourceManager;
  const auto *Construct =
      Result.Nodes.getNodeAs<CXXConstructExpr>(ConstructorCall);
  const auto *Type = Result.Nodes.getNodeAs<QualType>(PointerType);
  const auto *New = Result.Nodes.getNodeAs<CXXNewExpr>(NewExpression);

  if (New->getNumPlacementArgs() != 0)
    return;

  SourceLocation ConstructCallStart = Construct->getExprLoc();

  bool Invalid = false;
  StringRef ExprStr = Lexer::getSourceText(
      CharSourceRange::getCharRange(
          ConstructCallStart, Construct->getParenOrBraceRange().getBegin()),
      SM, LangOptions(), &Invalid);
  if (Invalid)
    return;

  auto Diag = diag(ConstructCallStart, "use %0 instead")
              << makeSmartPtrFunctionName;

  // Find the location of the template's left angle.
  size_t LAngle = ExprStr.find("<");
  SourceLocation ConstructCallEnd;
  if (LAngle == StringRef::npos) {
    // If the template argument is missing (because it is part of the alias)
    // we have to add it back.
    ConstructCallEnd = ConstructCallStart.getLocWithOffset(ExprStr.size());
    Diag << FixItHint::CreateInsertion(
        ConstructCallEnd, "<" + Type->getAsString(getLangOpts()) + ">");
  } else {
    ConstructCallEnd = ConstructCallStart.getLocWithOffset(LAngle);
  }

  Diag << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(ConstructCallStart, ConstructCallEnd),
      makeSmartPtrFunctionName);

  // If the smart_ptr is built with brace enclosed direct initialization, use
  // parenthesis instead.
  if (Construct->isListInitialization()) {
    SourceRange BraceRange = Construct->getParenOrBraceRange();
    Diag << FixItHint::CreateReplacement(
        CharSourceRange::getCharRange(
            BraceRange.getBegin(), BraceRange.getBegin().getLocWithOffset(1)),
        "(");
    Diag << FixItHint::CreateReplacement(
        CharSourceRange::getCharRange(BraceRange.getEnd(),
                                      BraceRange.getEnd().getLocWithOffset(1)),
        ")");
  }

  SourceLocation NewStart = New->getSourceRange().getBegin();
  SourceLocation NewEnd = New->getSourceRange().getEnd();
  switch (New->getInitializationStyle()) {
  case CXXNewExpr::NoInit: {
    Diag << FixItHint::CreateRemoval(SourceRange(NewStart, NewEnd));
    break;
  }
  case CXXNewExpr::CallInit: {
    SourceRange InitRange = New->getDirectInitRange();
    Diag << FixItHint::CreateRemoval(
        SourceRange(NewStart, InitRange.getBegin()));
    Diag << FixItHint::CreateRemoval(SourceRange(InitRange.getEnd(), NewEnd));
    break;
  }
  case CXXNewExpr::ListInit: {
    // Range of the substring that we do not want to remove.
    SourceRange InitRange;
    if (const auto *NewConstruct = New->getConstructExpr()) {
      // Direct initialization with initialization list.
      //   struct S { S(int x) {} };
      //   smart_ptr<S>(new S{5});
      // The arguments in the initialization list are going to be forwarded to
      // the constructor, so this has to be replaced with:
      //   struct S { S(int x) {} };
      //   std::make_smart_ptr<S>(5);
      InitRange = SourceRange(
          NewConstruct->getParenOrBraceRange().getBegin().getLocWithOffset(1),
          NewConstruct->getParenOrBraceRange().getEnd().getLocWithOffset(-1));
    } else {
      // Aggregate initialization.
      //   smart_ptr<Pair>(new Pair{first, second});
      // Has to be replaced with:
      //   smart_ptr<Pair>(Pair{first, second});
      InitRange = SourceRange(
          New->getAllocatedTypeSourceInfo()->getTypeLoc().getLocStart(),
          New->getInitializer()->getSourceRange().getEnd());
    }
    Diag << FixItHint::CreateRemoval(
        CharSourceRange::getCharRange(NewStart, InitRange.getBegin()));
    Diag << FixItHint::CreateRemoval(
        SourceRange(InitRange.getEnd().getLocWithOffset(1), NewEnd));
    break;
  }
  }
}

} // namespace modernize
} // namespace tidy
} // namespace clang
