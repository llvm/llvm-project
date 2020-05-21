//===--- InefficientAlgorithmCheck.cpp - clang-tidy------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InefficientAlgorithmCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace performance {

static bool areTypesCompatible(QualType Left, QualType Right) {
  if (const auto *LeftRefType = Left->getAs<ReferenceType>())
    Left = LeftRefType->getPointeeType();
  if (const auto *RightRefType = Right->getAs<ReferenceType>())
    Right = RightRefType->getPointeeType();
  return Left->getCanonicalTypeUnqualified() ==
         Right->getCanonicalTypeUnqualified();
}

void InefficientAlgorithmCheck::registerMatchers(MatchFinder *Finder) {
  const auto Algorithms =
      hasAnyName("::std::find", "::std::count", "::std::equal_range",
                 "::std::lower_bound", "::std::upper_bound");
  const auto ContainerMatcher = classTemplateSpecializationDecl(hasAnyName(
      "::std::set", "::std::map", "::std::multiset", "::std::multimap",
      "::std::unordered_set", "::std::unordered_map",
      "::std::unordered_multiset", "::std::unordered_multimap"));

  const auto Matcher = traverse(
      ast_type_traits::TK_AsIs,
      callExpr(
          callee(functionDecl(Algorithms)),
          hasArgument(
              0, cxxConstructExpr(has(ignoringParenImpCasts(cxxMemberCallExpr(
                     callee(cxxMethodDecl(hasName("begin"))),
                     on(declRefExpr(
                            hasDeclaration(decl().bind("IneffContObj")),
                            anyOf(hasType(ContainerMatcher.bind("IneffCont")),
                                  hasType(pointsTo(
                                      ContainerMatcher.bind("IneffContPtr")))))
                            .bind("IneffContExpr"))))))),
          hasArgument(
              1, cxxConstructExpr(has(ignoringParenImpCasts(cxxMemberCallExpr(
                     callee(cxxMethodDecl(hasName("end"))),
                     on(declRefExpr(
                         hasDeclaration(equalsBoundNode("IneffContObj"))))))))),
          hasArgument(2, expr().bind("AlgParam")),
          unless(isInTemplateInstantiation()))
          .bind("IneffAlg"));

  Finder->addMatcher(Matcher, this);
}

void InefficientAlgorithmCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *AlgCall = Result.Nodes.getNodeAs<CallExpr>("IneffAlg");
  const auto *IneffCont =
      Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("IneffCont");
  bool PtrToContainer = false;
  if (!IneffCont) {
    IneffCont =
        Result.Nodes.getNodeAs<ClassTemplateSpecializationDecl>("IneffContPtr");
    PtrToContainer = true;
  }
  const llvm::StringRef IneffContName = IneffCont->getName();
  const bool Unordered =
      IneffContName.find("unordered") != llvm::StringRef::npos;
  const bool Maplike = IneffContName.find("map") != llvm::StringRef::npos;

  // Store if the key type of the container is compatible with the value
  // that is searched for.
  QualType ValueType = AlgCall->getArg(2)->getType();
  QualType KeyType =
      IneffCont->getTemplateArgs()[0].getAsType().getCanonicalType();
  const bool CompatibleTypes = areTypesCompatible(KeyType, ValueType);

  // Check if the comparison type for the algorithm and the container matches.
  if (AlgCall->getNumArgs() == 4 && !Unordered) {
    const Expr *Arg = AlgCall->getArg(3);
    const QualType AlgCmp =
        Arg->getType().getUnqualifiedType().getCanonicalType();
    const unsigned CmpPosition =
        (IneffContName.find("map") == llvm::StringRef::npos) ? 1 : 2;
    const QualType ContainerCmp = IneffCont->getTemplateArgs()[CmpPosition]
                                      .getAsType()
                                      .getUnqualifiedType()
                                      .getCanonicalType();
    if (AlgCmp != ContainerCmp) {
      diag(Arg->getBeginLoc(),
           "different comparers used in the algorithm and the container");
      return;
    }
  }

  const auto *AlgDecl = AlgCall->getDirectCallee();
  if (!AlgDecl)
    return;

  if (Unordered && AlgDecl->getName().find("bound") != llvm::StringRef::npos)
    return;

  const auto *AlgParam = Result.Nodes.getNodeAs<Expr>("AlgParam");
  const auto *IneffContExpr = Result.Nodes.getNodeAs<Expr>("IneffContExpr");
  FixItHint Hint;

  SourceManager &SM = *Result.SourceManager;
  LangOptions LangOpts = getLangOpts();

  CharSourceRange CallRange =
      CharSourceRange::getTokenRange(AlgCall->getSourceRange());

  // FIXME: Create a common utility to extract a file range that the given token
  // sequence is exactly spelled at (without macro argument expansions etc.).
  // We can't use Lexer::makeFileCharRange here, because for
  //
  //   #define F(x) x
  //   x(a b c);
  //
  // it will return "x(a b c)", when given the range "a"-"c". It makes sense for
  // removals, but not for replacements.
  //
  // This code is over-simplified, but works for many real cases.
  if (SM.isMacroArgExpansion(CallRange.getBegin()) &&
      SM.isMacroArgExpansion(CallRange.getEnd())) {
    CallRange.setBegin(SM.getSpellingLoc(CallRange.getBegin()));
    CallRange.setEnd(SM.getSpellingLoc(CallRange.getEnd()));
  }

  if (!CallRange.getBegin().isMacroID() && !Maplike && CompatibleTypes) {
    StringRef ContainerText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(IneffContExpr->getSourceRange()), SM,
        LangOpts);
    StringRef ParamText = Lexer::getSourceText(
        CharSourceRange::getTokenRange(AlgParam->getSourceRange()), SM,
        LangOpts);
    std::string ReplacementText =
        (llvm::Twine(ContainerText) + (PtrToContainer ? "->" : ".") +
         AlgDecl->getName() + "(" + ParamText + ")")
            .str();
    Hint = FixItHint::CreateReplacement(CallRange, ReplacementText);
  }

  diag(AlgCall->getBeginLoc(),
       "this STL algorithm call should be replaced with a container method")
      << Hint;
}

} // namespace performance
} // namespace tidy
} // namespace clang
