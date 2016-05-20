//===-- FindAllMacros.h - find all macros -----------------------*- C++ -*-===//
//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_MACROS_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_MACROS_H

#include "SymbolInfo.h"
#include "SymbolReporter.h"
#include "clang/Lex/PPCallbacks.h"

namespace clang {
namespace find_all_symbols {

class HeaderMapCollector;

/// \brief A preprocessor that collects all macro symbols.
/// The contexts of a macro will be ignored since they are not available during
/// preprocessing period.
class FindAllMacros : public clang::PPCallbacks {
public:
  explicit FindAllMacros(SymbolReporter *Reporter, SourceManager *SM,
                         HeaderMapCollector *Collector = nullptr)
      : Reporter(Reporter), SM(SM), Collector(Collector) {}

  void MacroDefined(const Token &MacroNameTok,
                    const MacroDirective *MD) override;

private:
  // Reporter for SymbolInfo.
  SymbolReporter *const Reporter;
  SourceManager *const SM;
  // A remapping header file collector allowing clients to include a different
  // header.
  HeaderMapCollector *const Collector;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_FIND_ALL_MACROS_H
