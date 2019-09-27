//===--- HeaderSourceSwitch.h - ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_HEADERSOURCESWITCH_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_HEADERSOURCESWITCH_H

#include "ParsedAST.h"
#include "llvm/ADT/Optional.h"

namespace clang {
namespace clangd {

/// Given a header file, returns the best matching source file, and vice visa.
/// It only uses the filename heuristics to do the inference.
llvm::Optional<Path> getCorrespondingHeaderOrSource(
    const Path &OriginalFile,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS);

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_HEADERSOURCESWITCH_H
