//===--- FuzzySymbolIndex.cpp - Lookup symbols for autocomplete -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "FuzzySymbolIndex.h"
#include "llvm/Support/Regex.h"

using clang::find_all_symbols::SymbolAndSignals;
using llvm::StringRef;

namespace clang {
namespace include_fixer {
namespace {

class MemSymbolIndex : public FuzzySymbolIndex {
public:
  MemSymbolIndex(std::vector<SymbolAndSignals> Symbols) {
    for (auto &Symbol : Symbols) {
      auto Tokens = tokenize(Symbol.Symbol.getName());
      this->Symbols.emplace_back(
          StringRef(llvm::join(Tokens.begin(), Tokens.end(), " ")),
          std::move(Symbol));
    }
  }

  std::vector<SymbolAndSignals> search(StringRef Query) override {
    auto Tokens = tokenize(Query);
    llvm::Regex Pattern("^" + queryRegexp(Tokens));
    std::vector<SymbolAndSignals> Results;
    for (const Entry &E : Symbols)
      if (Pattern.match(E.first))
        Results.push_back(E.second);
    return Results;
  }

private:
  using Entry = std::pair<llvm::SmallString<32>, SymbolAndSignals>;
  std::vector<Entry> Symbols;
};

// Helpers for tokenize state machine.
enum TokenizeState {
  EMPTY,      // No pending characters.
  ONE_BIG,    // Read one uppercase letter, could be WORD or Word.
  BIG_WORD,   // Reading an uppercase WORD.
  SMALL_WORD, // Reading a lowercase word.
  NUMBER      // Reading a number.
};

enum CharType { UPPER, LOWER, DIGIT, MISC };
CharType classify(char c) {
  if (isupper(c))
    return UPPER;
  if (islower(c))
    return LOWER;
  if (isdigit(c))
    return DIGIT;
  return MISC;
}

} // namespace

std::vector<std::string> FuzzySymbolIndex::tokenize(StringRef Text) {
  std::vector<std::string> Result;
  // State describes the treatment of text from Start to I.
  // Once text is Flush()ed into Result, we're done with it and advance Start.
  TokenizeState State = EMPTY;
  size_t Start = 0;
  auto Flush = [&](size_t End) {
    if (State != EMPTY) {
      Result.push_back(Text.substr(Start, End - Start).lower());
      State = EMPTY;
    }
    Start = End;
  };
  for (size_t I = 0; I < Text.size(); ++I) {
    CharType Type = classify(Text[I]);
    if (Type == MISC)
      Flush(I);
    else if (Type == LOWER)
      switch (State) {
      case BIG_WORD:
        Flush(I - 1); // FOOBar: first token is FOO, not FOOB.
        LLVM_FALLTHROUGH;
      case ONE_BIG:
        State = SMALL_WORD;
        LLVM_FALLTHROUGH;
      case SMALL_WORD:
        break;
      default:
        Flush(I);
        State = SMALL_WORD;
      }
    else if (Type == UPPER)
      switch (State) {
      case ONE_BIG:
        State = BIG_WORD;
        LLVM_FALLTHROUGH;
      case BIG_WORD:
        break;
      default:
        Flush(I);
        State = ONE_BIG;
      }
    else if (Type == DIGIT && State != NUMBER) {
      Flush(I);
      State = NUMBER;
    }
  }
  Flush(Text.size());
  return Result;
}

std::string
FuzzySymbolIndex::queryRegexp(const std::vector<std::string> &Tokens) {
  std::string Result;
  for (size_t I = 0; I < Tokens.size(); ++I) {
    if (I)
      Result.append("[[:alnum:]]* ");
    for (size_t J = 0; J < Tokens[I].size(); ++J) {
      if (J)
        Result.append("([[:alnum:]]* )?");
      Result.push_back(Tokens[I][J]);
    }
  }
  return Result;
}

llvm::Expected<std::unique_ptr<FuzzySymbolIndex>>
FuzzySymbolIndex::createFromYAML(StringRef FilePath) {
  auto Buffer = llvm::MemoryBuffer::getFile(FilePath);
  if (!Buffer)
    return llvm::errorCodeToError(Buffer.getError());
  return llvm::make_unique<MemSymbolIndex>(
      find_all_symbols::ReadSymbolInfosFromYAML(Buffer.get()->getBuffer()));
}

} // namespace include_fixer
} // namespace clang
