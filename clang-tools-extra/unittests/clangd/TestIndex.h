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
#include "index/dex/DexIndex.h"
#include "index/dex/Iterator.h"
#include "index/dex/Token.h"
#include "index/dex/Trigram.h"

namespace clang {
namespace clangd {

// Creates Symbol instance and sets SymbolID to given QualifiedName.
Symbol symbol(llvm::StringRef QName);

// Bundles symbol pointers with the actual symbol slab the pointers refer to in
// order to ensure that the slab isn't destroyed while it's used by and index.
struct SlabAndPointers {
  SymbolSlab Slab;
  std::vector<const Symbol *> Pointers;
};

// Create a slab of symbols with the given qualified names as both IDs and
// names. The life time of the slab is managed by the returned shared pointer.
// If \p WeakSymbols is provided, it will be pointed to the managed object in
// the returned shared pointer.
std::shared_ptr<std::vector<const Symbol *>>
generateSymbols(std::vector<std::string> QualifiedNames,
                std::weak_ptr<SlabAndPointers> *WeakSymbols = nullptr);

// Create a slab of symbols with IDs and names [Begin, End], otherwise identical
// to the `generateSymbols` above.
std::shared_ptr<std::vector<const Symbol *>>
generateNumSymbols(int Begin, int End,
                   std::weak_ptr<SlabAndPointers> *WeakSymbols = nullptr);

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
