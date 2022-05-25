//===-- IndexHelpers.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_TESTINDEX_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTESTS_TESTINDEX_H

#include "index/Index.h"

namespace clang {
namespace clangd {

// Creates Symbol instance and sets SymbolID to given QualifiedName.
Symbol symbol(llvm::StringRef QName);

// Helpers to produce fake index symbols with proper SymbolID.
// USRFormat is a regex replacement string for the unqualified part of the USR.
Symbol sym(llvm::StringRef QName, index::SymbolKind Kind,
           llvm::StringRef USRFormat);
// Creats a function symbol assuming no function arg.
Symbol func(llvm::StringRef Name);
// Creates a class symbol.
Symbol cls(llvm::StringRef Name);
// Creates an enum symbol.
Symbol enm(llvm::StringRef Name);
// Creates a variable symbol.
Symbol var(llvm::StringRef Name);
// Creates a namespace symbol.
Symbol ns(llvm::StringRef Name);
// Create a C++20 concept symbol.
Symbol conceptSym(llvm::StringRef Name);

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
