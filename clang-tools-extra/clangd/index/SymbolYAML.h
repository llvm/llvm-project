//===--- SymbolYAML.h --------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SymbolYAML provides facilities to convert Symbol to YAML, and vice versa.
// The YAML format of Symbol is designed for simplicity and experiment, but
// isn't a suitable/efficient store.
//
// This is for **experimental** only. Don't use it in the production code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_FROM_YAML_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_FROM_YAML_H

#include "Index.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {

// Read symbols from a YAML-format string.
SymbolSlab symbolsFromYAML(llvm::StringRef YAMLContent);

// Read one symbol from a YAML-stream.
// The returned symbol is backed by Input.
Symbol SymbolFromYAML(llvm::yaml::Input &Input);

// Convert a single symbol to YAML-format string.
// The YAML result is safe to concatenate.
std::string SymbolToYAML(Symbol Sym);

// Convert symbols to a YAML-format string.
// The YAML result is safe to concatenate if you have multiple symbol slabs.
void SymbolsToYAML(const SymbolSlab &Symbols, llvm::raw_ostream &OS);

// Build an in-memory static index for global symbols from a symbol file.
// The size of global symbols should be relatively small, so that all symbols
// can be managed in memory.
std::unique_ptr<SymbolIndex> loadIndex(llvm::StringRef SymbolFilename,
                                       llvm::ArrayRef<std::string> URISchemes,
                                       bool UseDex = true);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_INDEX_SYMBOL_FROM_YAML_H
