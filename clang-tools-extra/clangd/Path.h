//===--- Path.h - Helper typedefs --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
