//===--- TestLexer.h - Format C++ code --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a TestLexer to create FormatTokens from strings.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_UNITTESTS_FORMAT_TESTLEXER_H
#define CLANG_UNITTESTS_FORMAT_TESTLEXER_H

#include "../../lib/Format/FormatTokenLexer.h"

#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

#include <numeric>
#include <ostream>

namespace clang {
namespace format {

typedef llvm::SmallVector<FormatToken *, 8> TokenList;

inline std::ostream &operator<<(std::ostream &Stream, const FormatToken &Tok) {
  Stream << "(" << Tok.Tok.getName() << ", \"" << Tok.TokenText.str() << "\")";
  return Stream;
}
inline std::ostream &operator<<(std::ostream &Stream, const TokenList &Tokens) {
  Stream << "{";
  for (size_t I = 0, E = Tokens.size(); I != E; ++I) {
    Stream << (I > 0 ? ", " : "") << *Tokens[I];
  }
  Stream << "}";
  return Stream;
}

inline TokenList uneof(const TokenList &Tokens) {
  assert(!Tokens.empty() && Tokens.back()->is(tok::eof));
  return TokenList(Tokens.begin(), std::prev(Tokens.end()));
}

inline std::string text(llvm::ArrayRef<FormatToken *> Tokens) {
  return std::accumulate(Tokens.begin(), Tokens.end(), std::string(),
                         [](const std::string &R, FormatToken *Tok) {
                           return (R + Tok->TokenText).str();
                         });
}

class TestLexer {
public:
  TestLexer(FormatStyle Style = getLLVMStyle())
      : Style(Style), SourceMgr("test.cpp", ""),
        IdentTable(getFormattingLangOpts(Style)) {}

  TokenList lex(llvm::StringRef Code) {
    Buffers.push_back(
        llvm::MemoryBuffer::getMemBufferCopy(Code, "<scratch space>"));
    clang::FileID FID = SourceMgr.get().createFileID(SourceManager::Unowned,
                                                     Buffers.back().get());
    FormatTokenLexer Lex(SourceMgr.get(), FID, 0, Style, Encoding, Allocator,
                         IdentTable);
    auto Result = Lex.lex();
    return TokenList(Result.begin(), Result.end());
  }

  FormatToken *id(llvm::StringRef Code) {
    auto Result = uneof(lex(Code));
    assert(Result.size() == 1U && "Code must expand to 1 token.");
    return Result[0];
  }

  FormatStyle Style;
  encoding::Encoding Encoding = encoding::Encoding_UTF8;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> Buffers;
  clang::SourceManagerForFile SourceMgr;
  llvm::SpecificBumpPtrAllocator<FormatToken> Allocator;
  IdentifierTable IdentTable;
};

} // namespace format
} // namespace clang

#endif // LLVM_CLANG_UNITTESTS_FORMAT_TEST_LEXER_H
