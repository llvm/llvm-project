//===--- CommentParser.h - Doxygen comment parser ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Doxygen comment parser.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_COMMENT_PARSER_H
#define LLVM_CLANG_AST_COMMENT_PARSER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/AST/CommentLexer.h"
#include "clang/AST/Comment.h"
#include "clang/AST/CommentSema.h"
#include "llvm/Support/Allocator.h"

namespace clang {
class SourceManager;

namespace comments {
class CommandTraits;

/// Doxygen comment parser.
class Parser {
  Parser(const Parser &) LLVM_DELETED_FUNCTION;
  void operator=(const Parser &) LLVM_DELETED_FUNCTION;

  friend class TextTokenRetokenizer;

  Lexer &L;

  Sema &S;

  /// Allocator for anything that goes into AST nodes.
  llvm::BumpPtrAllocator &Allocator;

  /// Source manager for the comment being parsed.
  const SourceManager &SourceMgr;

  DiagnosticsEngine &Diags;

  DiagnosticBuilder Diag(SourceLocation Loc, unsigned DiagID) {
    return Diags.Report(Loc, DiagID);
  }

  const CommandTraits &Traits;

  /// Current lookahead token.  We can safely assume that all tokens are from
  /// a single source file.
  Token Tok;

  /// A stack of additional lookahead tokens.
  SmallVector<Token, 8> MoreLATokens;

  void consumeToken() {
    if (MoreLATokens.empty())
      L.lex(Tok);
    else {
      Tok = MoreLATokens.back();
      MoreLATokens.pop_back();
    }
  }

  void putBack(const Token &OldTok) {
    MoreLATokens.push_back(Tok);
    Tok = OldTok;
  }

  void putBack(ArrayRef<Token> Toks) {
    if (Toks.empty())
      return;

    MoreLATokens.push_back(Tok);
    for (const Token *I = &Toks.back(),
         *B = &Toks.front();
         I != B; --I) {
      MoreLATokens.push_back(*I);
    }

    Tok = Toks[0];
  }

public:
  Parser(Lexer &L, Sema &S, llvm::BumpPtrAllocator &Allocator,
         const SourceManager &SourceMgr, DiagnosticsEngine &Diags,
         const CommandTraits &Traits);

  /// Parse arguments for \\param command.
  void parseParamCommandArgs(ParamCommandComment *PC,
                             TextTokenRetokenizer &Retokenizer);

  /// Parse arguments for \\tparam command.
  void parseTParamCommandArgs(TParamCommandComment *TPC,
                              TextTokenRetokenizer &Retokenizer);

  void parseBlockCommandArgs(BlockCommandComment *BC,
                             TextTokenRetokenizer &Retokenizer,
                             unsigned NumArgs);

  BlockCommandComment *parseBlockCommand();
  InlineCommandComment *parseInlineCommand();

  HTMLStartTagComment *parseHTMLStartTag();
  HTMLEndTagComment *parseHTMLEndTag();

  BlockContentComment *parseParagraphOrBlockCommand();

  VerbatimBlockComment *parseVerbatimBlock();
  VerbatimLineComment *parseVerbatimLine();
  BlockContentComment *parseBlockContent();
  FullComment *parseFullComment();
};

} // end namespace comments
} // end namespace clang

#endif

