//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of scandir.
///
//===----------------------------------------------------------------------===//

#include "src/dirent/scandir.h"

#include "hdr/func/free.h"
#include "hdr/func/malloc.h"
#include "hdr/types/size_t.h"
#include "hdr/types/struct_dirent.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/libc_errno.h"
#include "src/__support/CPP/vector.h"
#include "src/dirent/closedir.h"
#include "src/dirent/opendir.h"
#include "src/dirent/readdir.h"
#include "src/stdlib/malloc.h"
#include "src/stdlib/qsort_util.h"
#include "src/string/memcpy.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, scandir,
                   (const char *dir, struct dirent ***namelist,
                    __scandir_filter_t filter, __scandir_compare_t compare)) {
  DIR *dir_fd = LIBC_NAMESPACE::opendir(dir);
  if (dir_fd == nullptr) {
    // errno set by opendir
    return -1;
  }

  int saved_errno = 0;
  LIBC_NAMESPACE::cpp::vector<struct dirent*> entries;

  while (true) {
    libc_errno = 0;
    struct dirent *entry = LIBC_NAMESPACE::readdir(dir_fd);
    if (entry == nullptr) {
      // If the readdir call failed it set errno
      saved_errno = libc_errno;
      break;
    }

    // Note, filter may modify errno
    if (filter != nullptr && !filter(entry)) {
      continue;
    }

    // struct dirent contains an equivalent of flexible array memeber we
    // allocate with malloc and use d_reclen as size.
    struct dirent *new_entry = static_cast<struct dirent*>(::malloc(entry->d_reclen));
    if (new_entry == nullptr) {
      saved_errno = ENOMEM;
    }
    LIBC_NAMESPACE::memcpy(new_entry, entry, entry->d_reclen);

    if (!entries.push_back(new_entry)) {
      free(new_entry);
      saved_errno = ENOMEM;
      break;
    }
  }

  // Closedir may modify errno and set it to EBADF, which is not amongst
  // POSIX-defined error codes for scandir. So we ignore closedir's errno.
  LIBC_NAMESPACE::closedir(dir_fd);

  struct dirent **result = static_cast<struct dirent**>(
      ::malloc(entries.size() * sizeof(struct dirent *)));

  if (result == nullptr) {
    saved_errno = ENOMEM;
  }

  if (saved_errno != 0) {
    for (struct dirent *entry: entries) {
      ::free(entry);
    }
    libc_errno = saved_errno;
    return -1;
  }

  if (compare != nullptr) {
    auto cmp_fn = [compare](const void *a, const void *b) {
      auto left = static_cast<const struct dirent **>(const_cast<void *>(a));
      auto right = static_cast<const struct dirent **>(const_cast<void *>(b));
      return compare(left, right);
    };
    internal::unstable_sort(entries.data(), entries.size(), sizeof(struct dirent*), cmp_fn);
  }

  for (size_t i = 0; i < entries.size(); ++i) {
    result[i] = entries[i];
  }

  *namelist = result;
  return static_cast<int>(entries.size());
}

} // namespace LIBC_NAMESPACE_DECL
