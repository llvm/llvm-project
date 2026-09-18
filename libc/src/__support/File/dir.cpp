//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the platform independent Dir class.
///
//===----------------------------------------------------------------------===//

#include "src/__support/File/dir.h"

#include "hdr/errno_macros.h"
#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "include/llvm-libc-types/__scandir_compare_t.h"
#include "include/llvm-libc-types/__scandir_filter_t.h"
#include "src/__support/CPP/mutex.h" // lock_guard
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/vector.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/error_or.h"
#include "src/__support/macros/config.h"
#include "src/stdlib/qsort_util.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {

ErrorOr<Dir *> Dir::fdopen(int fd) {
  LIBC_NAMESPACE::AllocChecker ac;
  Dir *dir = new (ac) Dir(fd);
  if (!ac)
    return LIBC_NAMESPACE::Error(ENOMEM);
  return dir;
}

ErrorOr<Dir *> Dir::open(const char *path) {
  auto fd = platform_opendir(path);
  if (!fd)
    return LIBC_NAMESPACE::Error(fd.error());

  return Dir::fdopen(fd.value());
}

ErrorOr<struct dirent *> Dir::read() {
  cpp::lock_guard lock(mutex);
  if (readptr >= fillsize) {
    auto readsize = platform_fetch_dirents(fd, buffer);
    if (!readsize)
      return LIBC_NAMESPACE::Error(readsize.error());
    fillsize = readsize.value();
    readptr = 0;
  }
  if (fillsize == 0)
    return nullptr;

  cpp::span<uint8_t> buf_span(buffer, BUFSIZE);

  if (fillsize - readptr < sizeof(struct dirent))
    return Error(EIO);

  struct dirent *d =
      reinterpret_cast<struct dirent *>(buf_span.subspan(readptr).data());

  size_t reclen = platform_dir_reclen(d);

  if (reclen == 0 || readptr + reclen > fillsize)
    return Error(EIO);

  readptr += reclen;
  return d;
}

int Dir::close() {
  {
    cpp::lock_guard lock(mutex);
    int retval = platform_closedir(fd);
    if (retval != 0)
      return retval;
  }
  delete this;
  return 0;
}

ErrorOr<int> Dir::scan(const char *name, struct dirent ***namelist,
                       __scandir_filter_t filter, __scandir_compare_t compare) {
  auto res_open = Dir::open(name);
  if (!res_open) {
    return LIBC_NAMESPACE::Error(res_open.error());
  }
  Dir *dir = res_open.value();

  cpp::vector<struct dirent *> entries;
  int saved_errno = 0;

  while (true) {
    auto res_read = dir->read();
    if (!res_read) {
      saved_errno = res_read.error();
      break;
    }

    struct dirent *entry = res_read.value();
    if (entry == nullptr) {
      break;
    }

    // Note, filter may modify errno
    if (filter != nullptr && !filter(entry)) {
      continue;
    }

    // struct dirent contains an equivalent of a flexible array memeber, so
    // we can't use sizeof and d_reclen member is only available on Linux.
    size_t reclen = platform_dir_reclen(entry);

    struct dirent *new_entry = static_cast<struct dirent *>(::malloc(reclen));
    if (new_entry == nullptr) {
      saved_errno = ENOMEM;
    }
    inline_memcpy(new_entry, entry, reclen);

    if (!entries.push_back(new_entry)) {
      ::free(new_entry);
      saved_errno = ENOMEM;
      break;
    }
  }

  // Closedir may modify errno and set it to, e.g. EBADF, which is not amongst
  // POSIX-defined error codes for scandir.
  dir->close();

  struct dirent **result = static_cast<struct dirent **>(
      ::malloc(entries.size() * sizeof(struct dirent *)));

  if (result == nullptr) {
    saved_errno = ENOMEM;
  }

  if (saved_errno != 0) {
    for (struct dirent *entry : entries) {
      ::free(entry);
    }
    return LIBC_NAMESPACE::Error(saved_errno);
  }

  if (compare != nullptr) {
    auto cmp_fn = [compare](const void *a, const void *b) {
      auto left = static_cast<const struct dirent **>(const_cast<void *>(a));
      auto right = static_cast<const struct dirent **>(const_cast<void *>(b));
      return compare(left, right);
    };
    internal::unstable_sort(entries.data(), entries.size(),
                            sizeof(struct dirent *), cmp_fn);
  }

  for (size_t i = 0; i < entries.size(); ++i) {
    result[i] = entries[i];
  }

  *namelist = result;
  return static_cast<int>(entries.size());
}

} // namespace LIBC_NAMESPACE_DECL
