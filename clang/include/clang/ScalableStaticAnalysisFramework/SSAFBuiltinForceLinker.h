//===- SSAFBuiltinForceLinker.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file pulls in all built-in SSAF extractor and format registrations
/// by referencing their anchor symbols, preventing the static linker from
/// discarding the containing object files.
///
/// Include this header (with IWYU pragma: keep) in any translation unit that
/// must guarantee these registrations are active — typically the entry point
/// of a binary that uses clangScalableStaticAnalysisFrameworkCore.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINFORCELINKER_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINFORCELINKER_H

namespace clang::ssaf {

#define ANCHOR(NAME) extern const volatile int NAME;
#include "BuiltinAnchorSources.def"

// Force the linker to link in the built-in SSAF registrations.
[[maybe_unused]] static const int BuiltinAnchorDestination = [] {
  int AnchorSources[]{
#define ANCHOR(NAME) NAME,
#include "BuiltinAnchorSources.def"
  };

  int SomeUse = 0;
  for (int V : AnchorSources)
    SomeUse |= V;
  return SomeUse;
}();

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_SSAFBUILTINFORCELINKER_H
