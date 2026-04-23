//===-- Internal implementation for ftw/nftw --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_FTW_FTW_IMPL_H
#define LLVM_LIBC_SRC_FTW_FTW_IMPL_H

#include "src/__support/CPP/expected.h"
#include "src/__support/CPP/string.h"
#ifdef LIBC_FULL_BUILD
#include "include/llvm-libc-types/struct_FTW.h"
#include "include/llvm-libc-types/struct_stat.h"
#else
#include <ftw.h>
#include <sys/stat.h>
#endif

namespace LIBC_NAMESPACE_DECL {
namespace ftw_impl {

struct AncestorDir {
  dev_t Dev;
  ino_t Ino;
  AncestorDir *Parent;
};

using NftwFn = int (*)(const char *FilePath, const struct stat *StatBuf,
                       int TFlag, struct FTW *FtwBuf);

using FtwFn = int (*)(const char *FilePath, const struct stat *StatBuf,
                      int TFlag);

// Unified callback wrapper - uses a union to avoid virtual functions
struct CallbackWrapper {
  bool IsNftw;
  union {
    NftwFn NftwFnVal;
    FtwFn FtwFnVal;
  };

  LIBC_INLINE int call(const char *Path, const struct stat *Sb, int Type,
                       struct FTW *Ftwbuf) const {
    if (IsNftw)
      return NftwFnVal(Path, Sb, Type, Ftwbuf);
    else
      return FtwFnVal(Path, Sb, Type);
  }
};

// Main implementation function - defined in ftw_impl.cpp
// Returns the callback return value on success (which might be non-zero),
// or an unexpected errno on failure.
cpp::expected<int, int> doMergedFtw(const cpp::string &DirPath,
                                    const CallbackWrapper &Fn, int FdLimit,
                                    int Flags, int Level,
                                    unsigned long StartDevice,
                                    AncestorDir *Ancestors);

} // namespace ftw_impl
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_FTW_FTW_IMPL_H
