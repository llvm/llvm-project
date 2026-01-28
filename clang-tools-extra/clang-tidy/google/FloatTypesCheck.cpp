//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FloatTypesCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

namespace clang {

using namespace ast_matchers;

namespace {

AST_POLYMORPHIC_MATCHER(isValidAndNotInMacro,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(TypeLoc,
                                                        FloatingLiteral)) {
  const SourceLocation Loc = Node.getBeginLoc();
  return Loc.isValid() && !Loc.isMacroID();
}

AST_MATCHER(TypeLoc, isLongDoubleType) {
  TypeLoc TL = Node;
  if (const auto QualLoc = Node.getAs<QualifiedTypeLoc>())
    TL = QualLoc.getUnqualifiedLoc();

  const auto BuiltinLoc = TL.getAs<BuiltinTypeLoc>();
  if (!BuiltinLoc)
    return false;

  if (const auto *BT = BuiltinLoc.getTypePtr())
    return BT->getKind() == BuiltinType::LongDouble;
  return false;
}

AST_MATCHER(FloatingLiteral, isLongDoubleLiteral) {
  if (const auto *BT =
          dyn_cast_if_present<BuiltinType>(Node.getType().getTypePtr()))
    return BT->getKind() == BuiltinType::LongDouble;
  return false;
}

} // namespace

namespace tidy::google::runtime {

void RuntimeFloatCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(typeLoc(loc(realFloatingPointType()),
                             isValidAndNotInMacro(), isLongDoubleType())
                         .bind("longDoubleTypeLoc"),
                     this);
  Finder->addMatcher(floatLiteral(isValidAndNotInMacro(), isLongDoubleLiteral())
                         .bind("longDoubleFloatLiteral"),
                     this);
}

void RuntimeFloatCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *TL = Result.Nodes.getNodeAs<TypeLoc>("longDoubleTypeLoc")) {
    diag(TL->getBeginLoc(), "%0 type is not portable and should not be used")
        << TL->getType();
  }

  if (const auto *FL =
          Result.Nodes.getNodeAs<FloatingLiteral>("longDoubleFloatLiteral")) {
    diag(FL->getBeginLoc(), "%0 type from literal suffix 'L' is not portable "
                            "and should not be used")
        << FL->getType();
  }
}

} // namespace tidy::google::runtime

} // namespace clang
