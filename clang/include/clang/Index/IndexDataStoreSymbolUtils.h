//===--- IndexDataStoreSymbolUtils.h - Utilities for indexstore symbols ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXDATASTORESYMBOLUTILS_H
#define LLVM_CLANG_INDEX_INDEXDATASTORESYMBOLUTILS_H

#include "indexstore/indexstore.h"
#include "clang/Index/IndexSymbol.h"

namespace clang {
namespace index {

/// Map an indexstore_symbol_kind_t to a SymbolKind, handling unknown values.
SymbolKind getSymbolKind(indexstore_symbol_kind_t K);

SymbolSubKind getSymbolSubKind(indexstore_symbol_subkind_t K);

/// Map an indexstore_symbol_language_t to a SymbolLanguage, handling unknown
/// values.
SymbolLanguage getSymbolLanguage(indexstore_symbol_language_t L);

/// Map an indexstore representation to a SymbolPropertySet, handling
/// unknown values.
SymbolPropertySet getSymbolProperties(uint64_t Props);

/// Map an indexstore representation to a SymbolRoleSet, handling unknown
/// values.
SymbolRoleSet getSymbolRoles(uint64_t Roles);

/// Map a SymbolLanguage to a indexstore_symbol_language_t.
indexstore_symbol_kind_t getIndexStoreKind(SymbolKind K);

indexstore_symbol_subkind_t getIndexStoreSubKind(SymbolSubKind K);

/// Map a SymbolLanguage to a indexstore_symbol_language_t.
indexstore_symbol_language_t getIndexStoreLang(SymbolLanguage L);

/// Map a SymbolPropertySet to its indexstore representation.
uint64_t getIndexStoreProperties(SymbolPropertySet Props);

/// Map a SymbolRoleSet to its indexstore representation.
uint64_t getIndexStoreRoles(SymbolRoleSet Roles);

} // end namespace index
} // end namespace clang

#endif // LLVM_CLANG_INDEX_INDEXDATASTORESYMBOLUTILS_H
