//===--- SymbolOccurrenceFinder.h - Clang refactoring library -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides functionality for finding all occurrences of a USR in a
/// given AST.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_OCCURRENCE_FINDER_H
#define LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_OCCURRENCE_FINDER_H

#include "clang/AST/AST.h"
#include "clang/Tooling/Refactor/SymbolOperation.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {
namespace tooling {
namespace rename {

// FIXME: make this an AST matcher. Wouldn't that be awesome??? I agree!
std::vector<SymbolOccurrence>
findSymbolOccurrences(const SymbolOperation &Operation, Decl *Decl);

} // end namespace rename
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_OCCURRENCE_FINDER_H
