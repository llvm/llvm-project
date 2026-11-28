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

#include <atomic>

namespace lld {
namespace utils {

/// Symbols can be appended with "(.__uniq.xxxx)?(.llvm.yyyy)?(.Tgm)?" where
/// "xxxx" and "yyyy" are numbers that could change between builds, and .Tgm is
/// the global merge functions suffix
/// (see GlobalMergeFunc::MergingInstanceSuffix). We need to use the root symbol
/// name before this suffix so these symbols can be matched with profiles which
/// may have different suffixes.
llvm::StringRef getRootSymbol(llvm::StringRef Name);

/// Accesses `x` as an atomic, like std::atomic_ref in C++20. Any thread may use
/// this while others access `x` the same way, but every access to `x` that can
/// overlap one of those must go through it as well. Technically UB but safe in
/// practice. There's no other real option for emulating atomic_ref in C++17.
template <typename T> std::atomic<T> &atomicRef(T &x) {
  static_assert(sizeof(std::atomic<T>) == sizeof(T) &&
                    alignof(std::atomic<T>) == alignof(T) &&
                    std::atomic<T>::is_always_lock_free,
                "std::atomic<T> must have the same representation as T");
  return reinterpret_cast<std::atomic<T> &>(x);
}

template <typename T> const std::atomic<T> &atomicRef(const T &x) {
  return atomicRef(const_cast<T &>(x));
}
} // namespace utils
} // namespace lld

#endif
