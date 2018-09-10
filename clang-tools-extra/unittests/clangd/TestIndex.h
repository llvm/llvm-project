//===-- IndexHelpers.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_INDEXTESTCOMMON_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_INDEXTESTCOMMON_H

#include "index/Index.h"
#include "index/Merge.h"

namespace clang {
namespace clangd {

// Creates Symbol instance and sets SymbolID to given QualifiedName.
Symbol symbol(llvm::StringRef QName);

// Create a slab of symbols with the given qualified names as IDs and names.
SymbolSlab generateSymbols(std::vector<std::string> QualifiedNames);

// Create a slab of symbols with IDs and names [Begin, End].
SymbolSlab generateNumSymbols(int Begin, int End);

// Returns fully-qualified name out of given symbol.
std::string getQualifiedName(const Symbol &Sym);

// Performs fuzzy matching-based symbol lookup given a query and an index.
// Incomplete is set true if more items than requested can be retrieved, false
// otherwise.
std::vector<std::string> match(const SymbolIndex &I,
                               const FuzzyFindRequest &Req,
                               bool *Incomplete = nullptr);

// Returns qualified names of symbols with any of IDs in the index.
std::vector<std::string> lookup(const SymbolIndex &I,
                                llvm::ArrayRef<SymbolID> IDs);

} // namespace clangd
} // namespace clang

#endif
