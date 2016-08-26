//===--- tools/extra/clang-rename/USRLocFinder.h - Clang rename tool ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides functionality for finding all instances of a USR in a given
/// AST.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_LOC_FINDER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_LOC_FINDER_H

#include "clang/AST/AST.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace clang {
namespace rename {

// FIXME: make this an AST matcher. Wouldn't that be awesome??? I agree!
std::vector<SourceLocation>
getLocationsOfUSRs(const std::vector<std::string> &USRs,
                   llvm::StringRef PrevName, Decl *Decl);

} // namespace rename
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_LOC_FINDER_H
