//===--- VE.cpp - Implement VE target feature support ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements VE TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "VE.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"

using namespace clang;
using namespace clang::targets;

static constexpr int NumBuiltins =
    clang::VE::LastTSBuiltin - Builtin::FirstTSBuiltin;

static constexpr auto BuiltinStorage = Builtin::Storage<NumBuiltins>::Make(
#define BUILTIN CLANG_BUILTIN_STR_TABLE
#include "clang/Basic/BuiltinsVE.def"
    , {
#define BUILTIN CLANG_BUILTIN_ENTRY
#include "clang/Basic/BuiltinsVE.def"
      });

void VETargetInfo::getTargetDefines(const LangOptions &Opts,
                                    MacroBuilder &Builder) const {
  Builder.defineMacro("__ve", "1");
  Builder.defineMacro("__ve__", "1");
  Builder.defineMacro("__NEC__", "1");
  // FIXME: define __FAST_MATH__ 1 if -ffast-math is enabled
  // FIXME: define __OPTIMIZE__ n if -On is enabled
  // FIXME: define __VECTOR__ n 1 if automatic vectorization is enabled

  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4");
  Builder.defineMacro("__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8");
}

std::pair<const char *, ArrayRef<Builtin::Info>>
VETargetInfo::getTargetBuiltinStorage() const {
  return {BuiltinStorage.StringTable, BuiltinStorage.Infos};
}
