//===-- FindAllSymbols.h - find all symbols----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_SYMBOL_MATCHER_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_SYMBOL_MATCHER_H

#include "SymbolInfo.h"
#include "SymbolReporter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include <string>

namespace clang {
namespace find_all_symbols {

class HeaderMapCollector;

/// \brief FindAllSymbols collects all classes, free standing functions and
/// global variables with some extra information such as the path of the header
/// file, the namespaces they are contained in, the type of variables and the
/// parameter types of functions.
///
/// NOTE:
///   - Symbols declared in main files are not collected since they can not be
///   included.
///   - Member functions are not collected because accessing them must go
///   through the class. #include fixer only needs the class name to find
///   headers.
///
class FindAllSymbols : public ast_matchers::MatchFinder::MatchCallback {
public:
  explicit FindAllSymbols(SymbolReporter *Reporter,
                          HeaderMapCollector *Collector = nullptr)
      : Reporter(Reporter), Collector(Collector) {}

  void registerMatchers(ast_matchers::MatchFinder *MatchFinder);

  void run(const ast_matchers::MatchFinder::MatchResult &result) override;

protected:
  void onEndOfTranslationUnit() override;

private:
  // Current source file being processed, filled by first symbol found.
  std::string Filename;
  // Findings for the current source file, flushed on onEndOfTranslationUnit.
  SymbolInfo::SignalMap FileSymbols;
  // Reporter for SymbolInfo.
  SymbolReporter *const Reporter;
  // A remapping header file collector allowing clients include a different
  // header.
  HeaderMapCollector *const Collector;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_SYMBOL_MATCHER_H
