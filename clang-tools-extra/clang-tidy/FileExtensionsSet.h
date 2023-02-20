//===--- FileExtensionsSet.h - clang-tidy -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FILE_EXTENSIONS_SET_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FILE_EXTENSIONS_SET_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

namespace clang::tidy {
typedef llvm::SmallSet<llvm::StringRef, 5> FileExtensionsSet;
} // namespace clang::tidy

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_FILE_EXTENSIONS_SET_H
