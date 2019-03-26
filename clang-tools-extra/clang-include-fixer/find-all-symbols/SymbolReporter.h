//===--- SymbolReporter.h - Symbol Reporter ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
