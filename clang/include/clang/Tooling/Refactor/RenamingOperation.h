//===--- RenamingOperation.h - -----------------------------*- C++ -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_RENAMING_OPERATION_H
#define LLVM_CLANG_TOOLING_REFACTOR_RENAMING_OPERATION_H

#include "clang/Basic/LLVM.h"
#include "clang/Tooling/Refactor/SymbolName.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {

class IdentifierTable;

namespace tooling {

class SymbolOperation;

namespace rename {

/// Return true if the new name is a valid language identifier.
bool isNewNameValid(const SymbolName &NewName, bool IsSymbolObjCSelector,
                    IdentifierTable &IDs, const LangOptions &LangOpts);
bool isNewNameValid(const SymbolName &NewName, const SymbolOperation &Operation,
                    IdentifierTable &IDs, const LangOptions &LangOpts);

/// \brief Finds the set of new names that apply to the symbols in the given
/// \c SymbolOperation.
void determineNewNames(SymbolName NewName, const SymbolOperation &Operation,
                       SmallVectorImpl<SymbolName> &NewNames,
                       const LangOptions &LangOpts);

} // end namespace rename
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_RENAMING_OPERATION_H
