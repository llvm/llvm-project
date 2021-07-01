//===--- Features.h - Compile-time configuration ------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_FEATURES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_FEATURES_H
#include <string>

// Export constants like CLANGD_BUILD_XPC
#include "Features.inc"

namespace clang {
namespace clangd {

// Returns a version string for clangd, e.g. "clangd 10.0.0"
std::string versionString();

// Returns a string describing the compile-time configuration.
// e.g. mac+debug+asan+grpc
std::string featureString();

} // namespace clangd
} // namespace clang

#endif
