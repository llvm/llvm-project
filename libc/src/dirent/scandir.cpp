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

#include "hdr/types/size_t.h"
#include "hdr/types/struct_dirent.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/libc_errno.h"
#include "src/__support/CPP/vector.h"
#include "src/dirent/closedir.h"
#include "src/dirent/opendir.h"
#include "src/dirent/readdir.h"
#include "src/stdlib/qsort_util.h"
#include "src/string/memcpy.h"

namespace LIBC_NAMESPACE_DECL {

void free_entries(LIBC_NAMESPACE::cpp::vector<struct dirent *> &entries) {
  for (struct dirent *entry: entries) {
    delete entry;
  }
}

LLVM_LIBC_FUNCTION(int, scandir,
                   (const char *dir, struct dirent ***namelist,
                    __scandir_filter_t filter, __scandir_compare_t compare)) {
  DIR *dir_fd = LIBC_NAMESPACE::opendir(dir);
  if (dir_fd == nullptr) {
    // errno set by opendir
    return -1;
  }

  LIBC_NAMESPACE::cpp::vector<struct dirent*> entries;

  while (true) {
    libc_errno = 0;
    struct dirent *entry = LIBC_NAMESPACE::readdir(dir_fd);
    if (libc_errno != 0) {
      free_entries(entries);
      LIBC_NAMESPACE::closedir(dir_fd);
      // errno set by readdir
      return -1;
    }

    if (entry == nullptr) {
      break;
    }

    if (filter != nullptr && !filter(entry)) {
      continue;
    }

    AllocChecker ac;
    struct dirent *new_entry = new (ac) struct dirent;
    if (!ac || new_entry == NULL) {
      free_entries(entries);
      LIBC_NAMESPACE::closedir(dir_fd);
      libc_errno = ENOMEM;
      return -1;
    }

    LIBC_NAMESPACE::memcpy(new_entry, entry, sizeof(struct dirent));

    if (!entries.push_back(new_entry)) {
      free_entries(entries);
      LIBC_NAMESPACE::closedir(dir_fd);
      libc_errno = ENOMEM;
      return -1;
    }
  }

  if (compare != nullptr) {
    auto cmp_fn = [compare](const void *a, const void *b) {
      auto left = static_cast<const struct dirent **>(const_cast<void *>(a));
      auto right = static_cast<const struct dirent **>(const_cast<void *>(b));
      return compare(left, right);
    };
    internal::unstable_sort(entries.data(), entries.size(), sizeof(struct dirent*), cmp_fn);
  }

  AllocChecker ac;
  struct dirent **result = new (ac) struct dirent*[entries.size()];
  if (!ac) {
    free_entries(entries);
    LIBC_NAMESPACE::closedir(dir_fd);
    libc_errno = ENOMEM;
    return -1;
  }

  for (size_t i = 0; i < entries.size(); ++i) {
    result[i] = entries[i];
  }

  *namelist = result;

  LIBC_NAMESPACE::closedir(dir_fd);

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
