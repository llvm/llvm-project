//===--- Annotations.h - Annotated source code for tests ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A clangd-specific version of llvm/Testing/Support/Annotations.h, replaces
// offsets and offset-based ranges with types from the LSP protocol.
//===---------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_ANNOTATIONS_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_ANNOTATIONS_H

#include "Protocol.h"
#include "llvm/Testing/Support/Annotations.h"

namespace clang {
namespace clangd {

/// Same as llvm::Annotations, but adjusts functions to LSP-specific types for
/// positions and ranges.
class Annotations : public llvm::Annotations {
  using Base = llvm::Annotations;

public:
  using llvm::Annotations::Annotations;

  Position point(llvm::StringRef Name = "") const;
  std::vector<Position> points(llvm::StringRef Name = "") const;

  clangd::Range range(llvm::StringRef Name = "") const;
  std::vector<clangd::Range> ranges(llvm::StringRef Name = "") const;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANGD_ANNOTATIONS_H
