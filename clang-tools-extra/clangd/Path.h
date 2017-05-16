//===--- Path.h - Helper typedefs --------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATH_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_PATH_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang {
namespace clangd {

/// A typedef to represent a file path. Used solely for more descriptive
/// signatures.
using Path = std::string;
/// A typedef to represent a ref to file path. Used solely for more descriptive
/// signatures.
using PathRef = llvm::StringRef;

} // namespace clangd
} // namespace clang

#endif
