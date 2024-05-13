//===--- Feature.h - Compile-time configuration ------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file is not named "Features.h" because of a conflict with libstdc++:
// https://github.com/clangd/clangd/issues/835
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FEATURE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FEATURE_H
#include <string>

// Export constants like CLANGD_BUILD_XPC
#include "Features.inc"

namespace clang {
namespace clangd {

// Returns a version string for clangd, e.g. "clangd 10.0.0"
std::string versionString();

// Returns the platform triple for clangd, e.g. "x86_64-pc-linux-gnu"
// May include both the host and target triple if they differ.
std::string platformString();

// Returns a string describing the compile-time configuration.
// e.g. mac+debug+asan+grpc
std::string featureString();

} // namespace clangd
} // namespace clang

#endif
