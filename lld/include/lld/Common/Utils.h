//===- Utils.h ------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// The file declares utils functions that can be shared across archs.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_UTILS_H
#define LLD_UTILS_H

#include "llvm/ADT/StringRef.h"

namespace lld {
namespace utils {

/// Symbols can be appended with "(.__uniq.xxxx)?(.llvm.yyyy)?(.Tgm)?" where
/// "xxxx" and "yyyy" are numbers that could change between builds, and .Tgm is
/// the global merge functions suffix
/// (see GlobalMergeFunc::MergingInstanceSuffix). We need to use the root symbol
/// name before this suffix so these symbols can be matched with profiles which
/// may have different suffixes.
llvm::StringRef getRootSymbol(llvm::StringRef Name);
} // namespace utils
} // namespace lld

#endif
