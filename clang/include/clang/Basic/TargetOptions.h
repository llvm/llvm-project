//===--- TargetOptions.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the clang::TargetOptions class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_TARGETOPTIONS_H
#define LLVM_CLANG_BASIC_TARGETOPTIONS_H

#include "clang/Basic/OpenCLOptions.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Target/TargetOptions.h"
#include <string>
#include <vector>

namespace clang {

/// Options for controlling the target.
class TargetOptions {
public:
#define TYPED_TARGETOPT(Type, Name, Description) Type Name;
#include "clang/Basic/TargetOptions.def"
  TargetOptions() : ForceEnableInt128(false), NVPTXUseShortPointers(false) {}
};

}  // end namespace clang

#endif
