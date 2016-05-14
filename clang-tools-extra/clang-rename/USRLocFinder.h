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

#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace clang {

class Decl;
class SourceLocation;

namespace rename {

// FIXME: make this an AST matcher. Wouldn't that be awesome??? I agree!
std::vector<SourceLocation> getLocationsOfUSR(llvm::StringRef usr,
                                              llvm::StringRef PrevName,
                                              Decl *decl);
}
}

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_RENAME_USR_LOC_FINDER_H
