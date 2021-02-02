//===--- Path.cpp -------------------------------------------*- C++-*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Path.h"
namespace clang {
namespace clangd {

std::string maybeCaseFoldPath(PathRef Path) {
#if defined(_WIN32) || defined(__APPLE__)
  return Path.lower();
#else
  return std::string(Path);
#endif
}

bool pathEqual(PathRef A, PathRef B) {
#if defined(_WIN32) || defined(__APPLE__)
  return A.equals_lower(B);
#else
  return A == B;
#endif
}

} // namespace clangd
} // namespace clang
