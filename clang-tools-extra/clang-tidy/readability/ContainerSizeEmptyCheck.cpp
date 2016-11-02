//===--- ContainerSizeEmptyCheck.cpp - clang-tidy -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "ContainerSizeEmptyCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

ContainerSizeEmptyCheck::ContainerSizeEmptyCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void ContainerSizeEmptyCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus)
    return;

  const auto ValidContainer = cxxRecordDecl(isSameOrDerivedFrom(
      namedDecl(
          has(cxxMethodDecl(
                  isConst(), parameterCountIs(0), isPublic(), hasName("size"),
                  returns(qualType(isInteger(), unless(booleanType()))))
                  .bind("size")),
          has(cxxMethodDecl(isConst(), parameterCountIs(0), isPublic(),
                            hasName("empty"), returns(booleanType()))
                  .bind("empty")))
          .bind("container")));

  const auto WrongUse = anyOf(
      hasParent(binaryOperator(
                    matchers::isComparisonOperator(),
                    hasEitherOperand(ignoringImpCasts(anyOf(
                        integerLiteral(equals(1)), integerLiteral(equals(0))))))
                    .bind("SizeBinaryOp")),
      hasParent(implicitCastExpr(
          hasImplicitDestinationType(booleanType()),
          anyOf(
              hasParent(unaryOperator(hasOperatorName("!")).bind("NegOnSize")),
              anything()))),
      hasParent(explicitCastExpr(hasDestinationType(booleanType()))));

  Finder->addMatcher(
      cxxMemberCallExpr(on(expr(anyOf(hasType(ValidContainer),
                                      hasType(pointsTo(ValidContainer)),
                                      hasType(references(ValidContainer))))
                               .bind("STLObject")),
                        callee(cxxMethodDecl(hasName("size"))), WrongUse)
          .bind("SizeCallExpr"),
      this);
}

void ContainerSizeEmptyCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemberCall =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("SizeCallExpr");
  const auto *BinaryOp = Result.Nodes.getNodeAs<BinaryOperator>("SizeBinaryOp");
  const auto *E = Result.Nodes.getNodeAs<Expr>("STLObject");
  FixItHint Hint;
  std::string ReplacementText =
      Lexer::getSourceText(CharSourceRange::getTokenRange(E->getSourceRange()),
                           *Result.SourceManager, getLangOpts());
  if (E->getType()->isPointerType())
    ReplacementText += "->empty()";
  else
    ReplacementText += ".empty()";

  if (BinaryOp) { // Determine the correct transformation.
    bool Negation = false;
    const bool ContainerIsLHS =
        !llvm::isa<IntegerLiteral>(BinaryOp->getLHS()->IgnoreImpCasts());
    const auto OpCode = BinaryOp->getOpcode();
    uint64_t Value = 0;
    if (ContainerIsLHS) {
      if (const auto *Literal = llvm::dyn_cast<IntegerLiteral>(
              BinaryOp->getRHS()->IgnoreImpCasts()))
        Value = Literal->getValue().getLimitedValue();
      else
        return;
    } else {
      Value =
          llvm::dyn_cast<IntegerLiteral>(BinaryOp->getLHS()->IgnoreImpCasts())
              ->getValue()
              .getLimitedValue();
    }

    // Constant that is not handled.
    if (Value > 1)
      return;

    if (Value == 1 && (OpCode == BinaryOperatorKind::BO_EQ ||
                       OpCode == BinaryOperatorKind::BO_NE))
      return;

    // Always true, no warnings for that.
    if ((OpCode == BinaryOperatorKind::BO_GE && Value == 0 && ContainerIsLHS) ||
        (OpCode == BinaryOperatorKind::BO_LE && Value == 0 && !ContainerIsLHS))
      return;

    // Do not warn for size > 1, 1 < size, size <= 1, 1 >= size.
    if (Value == 1) {
      if ((OpCode == BinaryOperatorKind::BO_GT && ContainerIsLHS) ||
          (OpCode == BinaryOperatorKind::BO_LT && !ContainerIsLHS))
        return;
      if ((OpCode == BinaryOperatorKind::BO_LE && ContainerIsLHS) ||
          (OpCode == BinaryOperatorKind::BO_GE && !ContainerIsLHS))
        return;
    }

    if (OpCode == BinaryOperatorKind::BO_NE && Value == 0)
      Negation = true;
    if ((OpCode == BinaryOperatorKind::BO_GT ||
         OpCode == BinaryOperatorKind::BO_GE) &&
        ContainerIsLHS)
      Negation = true;
    if ((OpCode == BinaryOperatorKind::BO_LT ||
         OpCode == BinaryOperatorKind::BO_LE) &&
        !ContainerIsLHS)
      Negation = true;

    if (Negation)
      ReplacementText = "!" + ReplacementText;
    Hint = FixItHint::CreateReplacement(BinaryOp->getSourceRange(),
                                        ReplacementText);

  } else {
    // If there is a conversion above the size call to bool, it is safe to just
    // replace size with empty.
    if (const auto *UnaryOp =
            Result.Nodes.getNodeAs<UnaryOperator>("NegOnSize"))
      Hint = FixItHint::CreateReplacement(UnaryOp->getSourceRange(),
                                          ReplacementText);
    else
      Hint = FixItHint::CreateReplacement(MemberCall->getSourceRange(),
                                          "!" + ReplacementText);
  }

  diag(MemberCall->getLocStart(), "the 'empty' method should be used to check "
                                  "for emptiness instead of 'size'")
      << Hint;

  const auto *Container = Result.Nodes.getNodeAs<NamedDecl>("container");
  const auto *Empty = Result.Nodes.getNodeAs<FunctionDecl>("empty");

  diag(Empty->getLocation(), "method %0::empty() defined here",
       DiagnosticIDs::Note)
      << Container;
}

} // namespace readability
} // namespace tidy
} // namespace clang
