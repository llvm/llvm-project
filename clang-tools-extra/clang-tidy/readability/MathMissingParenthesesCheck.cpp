//===--- MathMissingParenthesesCheck.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MathMissingParenthesesCheck.h"
#include "../utils/ASTUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include <set>
#include <stack>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

void MathMissingParenthesesCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(binaryOperator(unless(hasParent(binaryOperator())),
                                    hasDescendant(binaryOperator()))
                         .bind("binOp"),
                     this);
}
static int precedenceCheck(const char op) {
  if (op == '/' || op == '*' || op == '%')
    return 5;

  else if (op == '+' || op == '-')
    return 4;

  else if (op == '&')
    return 3;
  else if (op == '^')
    return 2;

  else if (op == '|')
    return 1;

  else
    return 0;
}
static bool isOperand(const char c) {
  if (c >= 'a' && c <= 'z')
    return true;
  else if (c >= 'A' && c <= 'Z')
    return true;
  else if (c >= '0' && c <= '9')
    return true;
  else if (c == '$')
    return true;
  else
    return false;
}
static bool conditionForNegative(const std::string s, int i,
                                 const std::string CurStr) {
  if (CurStr[0] == '-') {
    if (i == 0) {
      return true;
    } else {
      while (s[i - 1] == ' ') {
        i--;
      }
      if (!isOperand(s[i - 1])) {
        return true;
      } else {
        return false;
      }
    }
  } else {
    return false;
  }
}
static std::string getOperationOrder(std::string s, std::set<char> &Operators) {
  std::stack<std::string> StackOne;
  std::string TempStr = "";
  for (int i = 0; i < s.length(); i++) {
    std::string CurStr = "";
    CurStr += s[i];
    if (CurStr == " ")
      continue;
    else {
      if (isOperand(CurStr[0]) || conditionForNegative(s, i, CurStr)) {
        while (i < s.length() && (isOperand(s[i]) || s[i] == '-')) {
          if (s[i] == '-') {
            TempStr += "$";
          } else {
            TempStr += CurStr;
          }
          i++;
          CurStr = s[i];
        }
        TempStr += " ";
      } else if (CurStr == "(") {
        StackOne.push("(");
      } else if (CurStr == ")") {
        while (StackOne.top() != "(") {
          TempStr += StackOne.top();
          StackOne.pop();
        }
        StackOne.pop();
      } else {
        while (!StackOne.empty() && precedenceCheck(CurStr[0]) <=
                                        precedenceCheck((StackOne.top())[0])) {
          TempStr += StackOne.top();
          StackOne.pop();
        }
        StackOne.push(CurStr);
      }
    }
  }
  while (!StackOne.empty()) {
    TempStr += StackOne.top();
    StackOne.pop();
  }
  std::stack<std::string> StackTwo;
  for (int i = 0; i < TempStr.length(); i++) {
    if (TempStr[i] == ' ')
      continue;
    else if (isOperand(TempStr[i])) {
      std::string CurStr = "";
      while (i < TempStr.length() && isOperand(TempStr[i])) {
        if (TempStr[i] == '$') {
          CurStr += "-";
        } else {
          CurStr += TempStr[i];
        }
        i++;
      }
      StackTwo.push(CurStr);
    } else {
      std::string OperandOne = StackTwo.top();
      StackTwo.pop();
      std::string OperandTwo = StackTwo.top();
      StackTwo.pop();
      Operators.insert(TempStr[i]);
      StackTwo.push("(" + OperandTwo + " " + TempStr[i] + " " + OperandOne +
                    ")");
    }
  }
  return StackTwo.top();
}
void MathMissingParenthesesCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  if (!BinOp)
    return;
  clang::SourceManager &SM = *Result.SourceManager;
  clang::LangOptions LO = Result.Context->getLangOpts();
  clang::CharSourceRange Range =
      clang::CharSourceRange::getTokenRange(BinOp->getSourceRange());
  std::string Expression = clang::Lexer::getSourceText(Range, SM, LO).str();
  std::set<char> Operators;
  std::string FinalExpression = getOperationOrder(Expression, Operators);
  if (Operators.size() > 1) {
    if (FinalExpression.length() > 2) {
      FinalExpression = FinalExpression.substr(1, FinalExpression.length() - 2);
    }
    diag(BinOp->getBeginLoc(),
         "add parantheses to clarify the precedence of operations")
        << FixItHint::CreateReplacement(Range, FinalExpression);
  }
}

} // namespace clang::tidy::readability
