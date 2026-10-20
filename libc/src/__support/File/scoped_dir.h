//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// RAII wrapper for Dir pointers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_SCOPED_DIR_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_SCOPED_DIR_H

#include "src/__support/File/dir.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

class ScopedDir {
  Dir *dir;

public:
  LIBC_INLINE explicit ScopedDir(Dir *d = nullptr) : dir(d) {}

  LIBC_INLINE ~ScopedDir() {
    if (dir != nullptr)
      dir->close();
  }

  LIBC_INLINE ScopedDir(const ScopedDir &) = delete;
  LIBC_INLINE ScopedDir &operator=(const ScopedDir &) = delete;

  LIBC_INLINE ScopedDir(ScopedDir &&other) : dir(other.dir) {
    other.dir = nullptr;
  }

  LIBC_INLINE ScopedDir &operator=(ScopedDir &&other) {
    if (this != &other) {
      if (dir != nullptr)
        dir->close();
      dir = other.dir;
      other.dir = nullptr;
    }
    return *this;
  }

  LIBC_INLINE int close() {
    if (dir == nullptr)
      return 0;
    Dir *d = dir;
    dir = nullptr;
    return d->close();
  }

  LIBC_INLINE Dir *get() const { return dir; }
  LIBC_INLINE Dir *operator->() const { return dir; }
  LIBC_INLINE explicit operator bool() const { return dir != nullptr; }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_SCOPED_DIR_H
