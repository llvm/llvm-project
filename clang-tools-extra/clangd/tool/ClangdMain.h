//===--- ClangdMain.h - clangd main function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_TOOL_CLANGDMAIN_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_TOOL_CLANGDMAIN_H

namespace clang {
namespace clangd {
// clangd main function (clangd server loop)
int clangdMain(int argc, char *argv[]);
} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_TOOL_CLANGDMAIN_H
