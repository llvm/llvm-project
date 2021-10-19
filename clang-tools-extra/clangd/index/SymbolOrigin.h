//===--- SymbolOrigin.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_ORIGIN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_ORIGIN_H

#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace clang {
namespace clangd {

// Describes the source of information about a symbol.
// Mainly useful for debugging, e.g. understanding code completion results.
// This is a bitfield as information can be combined from several sources.
enum class SymbolOrigin : uint8_t {
  Unknown = 0,
  AST = 1 << 0,        // Directly from the AST (indexes should not set this).
  Dynamic = 1 << 1,    // From the dynamic index of opened files.
  Static = 1 << 2,     // From the static, externally-built index.
  Merge = 1 << 3,      // A non-trivial index merge was performed.
  Identifier = 1 << 4, // Raw identifiers in file.
  Remote = 1 << 5,     // Remote index.
  // Remaining bits reserved for index implementations.
};

inline SymbolOrigin operator|(SymbolOrigin A, SymbolOrigin B) {
  return static_cast<SymbolOrigin>(static_cast<uint8_t>(A) |
                                   static_cast<uint8_t>(B));
}
inline SymbolOrigin &operator|=(SymbolOrigin &A, SymbolOrigin B) {
  return A = A | B;
}
inline SymbolOrigin operator&(SymbolOrigin A, SymbolOrigin B) {
  return static_cast<SymbolOrigin>(static_cast<uint8_t>(A) &
                                   static_cast<uint8_t>(B));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &, SymbolOrigin);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_ORIGIN_H
