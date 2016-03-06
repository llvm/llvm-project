//===--- FixItHintUtils.cpp - clang-tidy-----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FixItHintUtils.h"
#include "LexerUtils.h"
#include "clang/AST/ASTContext.h"

namespace clang {
namespace tidy {
namespace utils {
namespace create_fix_it {

FixItHint changeVarDeclToReference(const VarDecl &Var, ASTContext &Context) {
  SourceLocation AmpLocation = Var.getLocation();
  auto Token = lexer_utils::getPreviousNonCommentToken(Context, AmpLocation);
  if (!Token.is(tok::unknown))
    AmpLocation = Lexer::getLocForEndOfToken(Token.getLocation(), 0,
                                             Context.getSourceManager(),
                                             Context.getLangOpts());
  return FixItHint::CreateInsertion(AmpLocation, "&");
}

FixItHint changeVarDeclToConst(const VarDecl &Var) {
  return FixItHint::CreateInsertion(Var.getTypeSpecStartLoc(), "const ");
}

} // namespace create_fix_it
} // namespace utils
} // namespace tidy
} // namespace clang
