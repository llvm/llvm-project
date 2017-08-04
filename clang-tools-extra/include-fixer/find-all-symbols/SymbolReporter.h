//===--- SymbolReporter.h - Symbol Reporter ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_SYMBOL_REPORTER_H
#define LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_SYMBOL_REPORTER_H

#include "SymbolInfo.h"

namespace clang {
namespace find_all_symbols {

/// \brief An interface for classes that collect symbols.
class SymbolReporter {
public:
  virtual ~SymbolReporter() = default;

  virtual void reportSymbols(llvm::StringRef FileName,
                             const SymbolInfo::SignalMap &Symbols) = 0;
};

} // namespace find_all_symbols
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_FIND_ALL_SYMBOLS_SYMBOL_REPORTER_H
