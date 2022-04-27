//===--- Lex.cpp - extract token stream from source code ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Token.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/LiteralSupport.h"

namespace clang {
namespace pseudo {

TokenStream lex(const std::string &Code, const clang::LangOptions &LangOpts) {
  clang::SourceLocation Start;
  // Tokenize using clang's lexer in raw mode.
  // std::string guarantees null-termination, which the lexer needs.
  clang::Lexer Lexer(Start, LangOpts, Code.data(), Code.data(),
                     Code.data() + Code.size());
  Lexer.SetCommentRetentionState(true);

  TokenStream Result;
  clang::Token CT;
  unsigned LastOffset = 0;
  unsigned Line = 0;
  unsigned Indent = 0;
  for (Lexer.LexFromRawLexer(CT); CT.getKind() != clang::tok::eof;
       Lexer.LexFromRawLexer(CT)) {
    unsigned Offset =
        CT.getLocation().getRawEncoding() - Start.getRawEncoding();

    Token Tok;
    Tok.Data = &Code[Offset];
    Tok.Length = CT.getLength();
    Tok.Kind = CT.getKind();

    // Update current line number and indentation from raw source code.
    unsigned NewLineStart = 0;
    for (unsigned I = LastOffset; I < Offset; ++I) {
      if (Code[I] == '\n') {
        NewLineStart = I + 1;
        ++Line;
      }
    }
    if (NewLineStart || !LastOffset) {
      Indent = 0;
      for (char C : StringRef(Code).slice(NewLineStart, Offset)) {
        if (C == ' ')
          ++Indent;
        else if (C == '\t')
          Indent += 8;
        else
          break;
      }
    }
    Tok.Indent = Indent;
    Tok.Line = Line;

    if (CT.isAtStartOfLine())
      Tok.setFlag(LexFlags::StartsPPLine);
    if (CT.needsCleaning() || CT.hasUCN())
      Tok.setFlag(LexFlags::NeedsCleaning);

    Result.push(Tok);
    LastOffset = Offset;
  }
  Result.finalize();
  return Result;
}

TokenStream cook(const TokenStream &Code, const LangOptions &LangOpts) {
  auto CleanedStorage = std::make_shared<llvm::BumpPtrAllocator>();
  clang::IdentifierTable Identifiers(LangOpts);
  TokenStream Result(CleanedStorage);

  for (auto Tok : Code.tokens()) {
    if (Tok.flag(LexFlags::NeedsCleaning)) {
      // Remove escaped newlines and trigraphs.
      llvm::SmallString<64> CleanBuffer;
      const char *Pos = Tok.text().begin();
      while (Pos < Tok.text().end()) {
        unsigned CharSize = 0;
        CleanBuffer.push_back(
            clang::Lexer::getCharAndSizeNoWarn(Pos, CharSize, LangOpts));
        assert(CharSize != 0 && "no progress!");
        Pos += CharSize;
      }
      // Remove universal character names (UCN).
      llvm::SmallString<64> UCNBuffer;
      clang::expandUCNs(UCNBuffer, CleanBuffer);

      llvm::StringRef Text = llvm::StringRef(UCNBuffer).copy(*CleanedStorage);
      Tok.Data = Text.data();
      Tok.Length = Text.size();
      Tok.Flags &= ~static_cast<decltype(Tok.Flags)>(LexFlags::NeedsCleaning);
    }

    if (Tok.Kind == tok::raw_identifier) {
      // Cook raw_identifiers into identifier, keyword, etc.
      Tok.Kind = Identifiers.get(Tok.text()).getTokenID();
    } else if (Tok.Kind == tok::greatergreater) {
      // Split the greatergreater token.
      // FIXME: split lessless token to support Cuda triple angle brackets <<<.
      assert(Tok.text() == ">>");
      Tok.Kind = tok::greater;
      Tok.Length = 1;
      Result.push(Tok);
      // Line is wrong if the first greater is followed by an escaped newline!
      Tok.Data = Tok.text().data() + 1;
    }

    Result.push(std::move(Tok));
  }

  Result.finalize();
  return Result;
}

} // namespace pseudo
} // namespace clang
