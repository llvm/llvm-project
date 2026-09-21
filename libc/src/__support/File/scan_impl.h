//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Templated implementation of the scandir logic for dependency injection.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FILE_SCAN_IMPL_H
#define LLVM_LIBC_SRC___SUPPORT_FILE_SCAN_IMPL_H

#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "hdr/types/struct_dirent.h"
#include "include/llvm-libc-types/__scandir_compare_t.h"
#include "include/llvm-libc-types/__scandir_filter_t.h"
#include "src/__support/CPP/vector.h"
#include "src/__support/File/dir.h"
#include "src/stdlib/qsort_util.h"
#include "src/string/memory_utils/inline_memcpy.h"

namespace LIBC_NAMESPACE_DECL {
namespace internal {

template <typename DirType>
ErrorOr<int> scan_impl(const char *name, struct dirent ***namelist,
                       __scandir_filter_t filter, __scandir_compare_t compare) {
  auto res_open = DirType::open(name);
  if (!res_open) {
    return LIBC_NAMESPACE::Error(res_open.error());
  }
  DirType *dir = res_open.value();

  cpp::vector<struct dirent *> entries;

  auto free_entries = [&entries]() {
    for (struct dirent *entry : entries) {
      ::free(entry);
    }
  };

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
      break;
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

  if (saved_errno != 0) {
    free_entries();
    return LIBC_NAMESPACE::Error(saved_errno);
  }

  size_t alloc_size = entries.size() * sizeof(struct dirent *); 
  // The filter may have filtered out all entries. We'd like to avoid
  // malloc(0) in this instance as its exact semantics might be
  // implementation-dependent.
  if (alloc_size == 0) {
    alloc_size = sizeof(struct dirent *);
  }

  struct dirent **result = static_cast<struct dirent **>(
      ::malloc(alloc_size));

  if (result == nullptr) {
    free_entries();
    return LIBC_NAMESPACE::Error(ENOMEM);
  }

  if (compare != nullptr && !entries.empty()) {
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

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_FILE_SCAN_IMPL_H
