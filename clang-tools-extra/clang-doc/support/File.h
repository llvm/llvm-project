//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains file I/O utility functions used in clang-doc,
/// such as creating directories and writing output files.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_FILE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_FILE_H

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"

namespace clang {
namespace doc {

llvm::Error copyFile(llvm::StringRef FilePath, llvm::StringRef OutDirectory);

llvm::SmallString<128> computeRelativePath(llvm::StringRef Destination,
                                           llvm::StringRef Origin);

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_FILE_H
