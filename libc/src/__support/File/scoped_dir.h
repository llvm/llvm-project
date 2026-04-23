//===-- A scoped Dir wrapper for RAII directory handling -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_SCOPED_DIR_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_SCOPED_DIR_H

#include "src/__support/File/dir.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// RAII wrapper for Dir that automatically closes the directory on destruction.
// Usage:
//   auto result = Dir::open(path);
//   if (!result) { handle error }
//   ScopedDir dir(result.value());
//   // dir automatically closes when it goes out of scope
class ScopedDir {
  Dir *dir = nullptr;

public:
  LIBC_INLINE ScopedDir() = default;
  LIBC_INLINE explicit ScopedDir(Dir *d) : dir(d) {}

  LIBC_INLINE ~ScopedDir() {
    if (dir)
      dir->close();
  }

  // Move-only type
  LIBC_INLINE ScopedDir(ScopedDir &&other) : dir(other.dir) {
    other.dir = nullptr;
  }
  LIBC_INLINE ScopedDir &operator=(ScopedDir &&other) {
    if (this != &other) {
      if (dir)
        dir->close();
      dir = other.dir;
      other.dir = nullptr;
    }
    return *this;
  }

  // Non-copyable
  ScopedDir(const ScopedDir &) = delete;
  ScopedDir &operator=(const ScopedDir &) = delete;

  LIBC_INLINE Dir *operator->() { return dir; }
  LIBC_INLINE Dir &operator*() { return *dir; }
  LIBC_INLINE explicit operator bool() const { return dir != nullptr; }
  LIBC_INLINE Dir *get() { return dir; }

  // Release ownership without closing
  LIBC_INLINE Dir *release() {
    Dir *tmp = dir;
    dir = nullptr;
    return tmp;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_SCOPED_DIR_H
