//===--- FuzzySymbolIndex.h - Lookup symbols for autocomplete ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FUZZY_SYMBOL_INDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FUZZY_SYMBOL_INDEX_H

#include "SymbolIndex.h"
#include "find-all-symbols/SymbolInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace clang {
namespace include_fixer {

// A FuzzySymbolIndex retrieves top-level symbols matching a query string.
//
// It refines the contract of SymbolIndex::search to do fuzzy matching:
// - symbol names are tokenized: "unique ptr", "string ref".
// - query must match prefixes of symbol tokens: [upt]
// - if the query has multiple tokens, splits must match: [StR], not [STr].
// Helpers for tokenization and regex matching are provided.
//
// Implementations may choose to truncate results, refuse short queries, etc.
class FuzzySymbolIndex : public SymbolIndex {
public:
  // Loads the specified include-fixer database and returns an index serving it.
  static llvm::Expected<std::unique_ptr<FuzzySymbolIndex>>
  createFromYAML(llvm::StringRef File);

  // Helpers for implementing indexes:

  // Transforms a symbol name or query into a sequence of tokens.
  // - URLHandlerCallback --> [url, handler, callback]
  // - snake_case11 --> [snake, case, 11]
  // - _WTF$ --> [wtf]
  static std::vector<std::string> tokenize(llvm::StringRef Text);

  // Transforms query tokens into an unanchored regexp to match symbol tokens.
  // - [fe f] --> /f(\w* )?e\w* f/, matches [fee fie foe].
  static std::string queryRegexp(const std::vector<std::string> &Tokens);
};

} // namespace include_fixer
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_INCLUDE_FIXER_FUZZY_SYMBOL_INDEX_H
