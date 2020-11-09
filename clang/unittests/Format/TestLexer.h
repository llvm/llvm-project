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
#include "../../lib/Format/TokenAnalyzer.h"
#include "../../lib/Format/TokenAnnotator.h"
#include "../../lib/Format/UnwrappedLineParser.h"

#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

#include <numeric>
#include <ostream>

namespace clang {
namespace format {

typedef llvm::SmallVector<FormatToken *, 8> TokenList;

inline std::ostream &operator<<(std::ostream &Stream, const FormatToken &Tok) {
  Stream << "(" << Tok.Tok.getName() << ", \"" << Tok.TokenText.str() << "\" , "
         << getTokenTypeName(Tok.getType()) << ")";
  return Stream;
}
inline std::ostream &operator<<(std::ostream &Stream, const TokenList &Tokens) {
  Stream << "{";
  for (size_t I = 0, E = Tokens.size(); I != E; ++I) {
    Stream << (I > 0 ? ", " : "") << *Tokens[I];
  }
  Stream << "} (" << Tokens.size() << " tokens)";
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

class TestLexer : public UnwrappedLineConsumer {
public:
  TestLexer(llvm::SpecificBumpPtrAllocator<FormatToken> &Allocator,
            std::vector<std::unique_ptr<llvm::MemoryBuffer>> &Buffers,
            FormatStyle Style = getLLVMStyle())
      : Allocator(Allocator), Buffers(Buffers), Style(Style),
        SourceMgr("test.cpp", ""), IdentTable(getFormattingLangOpts(Style)) {}

  TokenList lex(llvm::StringRef Code) {
    FormatTokenLexer Lex = getNewLexer(Code);
    ArrayRef<FormatToken *> Result = Lex.lex();
    return TokenList(Result.begin(), Result.end());
  }

  TokenList annotate(llvm::StringRef Code) {
    FormatTokenLexer Lex = getNewLexer(Code);
    auto Tokens = Lex.lex();
    UnwrappedLineParser Parser(Style, Lex.getKeywords(), 0, Tokens, *this);
    Parser.parse();
    TokenAnnotator Annotator(Style, Lex.getKeywords());
    for (auto &Line : UnwrappedLines) {
      AnnotatedLine Annotated(Line);
      Annotator.annotate(Annotated);
      Annotator.calculateFormattingInformation(Annotated);
    }
    UnwrappedLines.clear();
    return TokenList(Tokens.begin(), Tokens.end());
  }

  FormatToken *id(llvm::StringRef Code) {
    auto Result = uneof(lex(Code));
    assert(Result.size() == 1U && "Code must expand to 1 token.");
    return Result[0];
  }

protected:
  void consumeUnwrappedLine(const UnwrappedLine &TheLine) override {
    UnwrappedLines.push_back(TheLine);
  }
  void finishRun() override {}

  FormatTokenLexer getNewLexer(StringRef Code) {
    Buffers.push_back(
        llvm::MemoryBuffer::getMemBufferCopy(Code, "<scratch space>"));
    clang::FileID FID =
        SourceMgr.get().createFileID(Buffers.back()->getMemBufferRef());
    return FormatTokenLexer(SourceMgr.get(), FID, 0, Style, Encoding, Allocator,
                            IdentTable);
  }

public:
  llvm::SpecificBumpPtrAllocator<FormatToken>& Allocator;
  std::vector<std::unique_ptr<llvm::MemoryBuffer>>& Buffers;
  FormatStyle Style;
  encoding::Encoding Encoding = encoding::Encoding_UTF8;
  clang::SourceManagerForFile SourceMgr;
  IdentifierTable IdentTable;
  SmallVector<UnwrappedLine, 16> UnwrappedLines;
};

} // namespace format
} // namespace clang

#endif // LLVM_CLANG_UNITTESTS_FORMAT_TEST_LEXER_H
