//===--- Options.h - Option info & table ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_OPTIONS_H
#define LLVM_CLANG_DRIVER_OPTIONS_H

#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"

namespace clang {
namespace driver {

namespace options {
/// Flags specifically for clang options.  Must not overlap with
/// llvm::opt::DriverFlag.
enum ClangFlags {
  NoXarchOption = (1 << 4),
  LinkerInput = (1 << 5),
  NoArgumentUnused = (1 << 6),
  Unsupported = (1 << 7),
  CoreOption = (1 << 8),
  CLOption = (1 << 9),
  CC1Option = (1 << 10),
  CC1AsOption = (1 << 11),
  NoDriverOption = (1 << 12),
  LinkOption = (1 << 13),
  FlangOption = (1 << 14),
  FC1Option = (1 << 15),
  FlangOnlyOption = (1 << 16),
  DXCOption = (1 << 17),
  CLDXCOption = (1 << 18),
  Ignored = (1 << 19),
  TargetSpecific = (1 << 20),
};

// Flags specifically for clang option visibility. We alias DefaultVis to
// ClangOption, because "DefaultVis" is confusing in Options.td, which is used
// for multiple drivers (clang, cl, flang, etc).
enum ClangVisibility {
  ClangOption = llvm::opt::DefaultVis,
};

enum ID {
    OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "clang/Driver/Options.inc"
    LastOption
#undef OPTION
  };
}

const llvm::opt::OptTable &getDriverOptTable();
}
}

#endif
