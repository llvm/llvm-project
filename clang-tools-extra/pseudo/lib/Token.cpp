//===--- Token.cpp - Tokens and token streams in the pseudoparser ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Token.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"

namespace clang {
namespace pseudo {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Token &T) {
  OS << llvm::formatv("{0} {1}:{2} ", clang::tok::getTokenName(T.Kind), T.Line,
                      T.Indent);
  OS << '"';
  llvm::printEscapedString(T.text(), OS);
  OS << '"';
  if (T.Flags)
    OS << llvm::format(" flags=%x", T.Flags);
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const TokenStream &TS) {
  OS << "Index               Kind    Line  Text\n";
  for (const auto &T : TS.tokens()) {
    OS << llvm::format("%5d:  %16s %4d:%-2d  ", TS.index(T),
                       clang::tok::getTokenName(T.Kind), T.Line, T.Indent);
    OS << '"';
    llvm::printEscapedString(T.text(), OS);
    OS << '"';
    if (T.Flags)
      OS << llvm::format("  flags=%x", T.Flags);
    OS << '\n';
  }
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Token::Range &R) {
  OS << llvm::formatv("[{0},{1})", R.Begin, R.End);
  return OS;
}

TokenStream::TokenStream(std::shared_ptr<void> Payload)
    : Payload(std::move(Payload)) {
  Storage.emplace_back();
  Storage.back().Kind = clang::tok::eof;
}

void TokenStream::finalize() {
  assert(!isFinalized());
  unsigned LastLine = Storage.back().Line;
  Storage.emplace_back();
  Storage.back().Kind = tok::eof;
  Storage.back().Line = LastLine + 1;

  Tokens = Storage;
  Tokens = Tokens.drop_front().drop_back();
}

bool TokenStream::isFinalized() const {
  assert(!Storage.empty() && Storage.front().Kind == tok::eof);
  if (Storage.size() == 1)
    return false;
  return Storage.back().Kind == tok::eof;
}

void TokenStream::print(llvm::raw_ostream &OS) const {
  bool FirstToken = true;
  unsigned LastLine = -1;
  StringRef LastText;
  for (const auto &T : tokens()) {
    StringRef Text = T.text();
    if (FirstToken) {
      FirstToken = false;
    } else if (T.Line == LastLine) {
      if (LastText.data() + LastText.size() != Text.data())
        OS << ' ';
    } else {
      OS << '\n';
      OS.indent(T.Indent);
    }
    OS << Text;
    LastLine = T.Line;
    LastText = Text;
  }
  if (!FirstToken)
    OS << '\n';
}

TokenStream stripComments(const TokenStream &Input) {
  TokenStream Out;
  for (const Token &T : Input.tokens()) {
    if (T.Kind == tok::comment)
      continue;
    Out.push(T);
  }
  Out.finalize();
  return Out;
}

} // namespace pseudo
} // namespace clang
