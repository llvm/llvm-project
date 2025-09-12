//===- ForceLinker.h ----------------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTEST_FEATURE_MODULES_FORCE_LINKER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_UNITTEST_FEATURE_MODULES_FORCE_LINKER_H

#include "llvm/Support/Compiler.h"

namespace clang::clangd {
extern volatile int DummyFeatureModuleAnchorSource;
static int LLVM_ATTRIBUTE_UNUSED DummyFeatureModuleAnchorDestination =
    DummyFeatureModuleAnchorSource;
} // namespace clang::clangd

#endif
