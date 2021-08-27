//===--- SymbolOccurrenceFinder.h - Clang refactoring library -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
std::vector<OldSymbolOccurrence>
findSymbolOccurrences(const SymbolOperation &Operation, Decl *Decl);

} // end namespace rename
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_SYMBOL_OCCURRENCE_FINDER_H
