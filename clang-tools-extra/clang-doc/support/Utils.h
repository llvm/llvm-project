//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_FILE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_FILE_H

#include "../Representation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

/// Appends \p Path to \p Base and returns the appended path.
llvm::SmallString<128> appendPathNative(llvm::StringRef Base,
                                        llvm::StringRef Path);

/// Appends \p Path to \p Base and returns the appended path in posix style.
llvm::SmallString<128> appendPathPosix(llvm::StringRef Base,
                                       llvm::StringRef Path);

void getMustacheHtmlFiles(llvm::StringRef AssetsPath,
                          clang::doc::ClangDocContext &CDCtx);

#endif
