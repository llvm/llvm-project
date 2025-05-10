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

llvm::SmallString<128> appendPathNative(llvm::StringRef Path,
                                        llvm::StringRef Asset);

void getMustacheHtmlFiles(llvm::StringRef AssetsPath,
                          clang::doc::ClangDocContext &CDCtx);

#endif
