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
    if (entry == nullptr) {
      // If the readdir call failed it set errno
      break;
    }

    if (filter != nullptr && !filter(entry)) {
      continue;
    }

    AllocChecker ac;
    struct dirent *new_entry = new (ac) struct dirent;
    if (!ac || new_entry == NULL) {
      libc_errno = ENOMEM;
      break;
    }

    // struct dirent contains an equivalent of flexible array memeber,
    // which makes sizeof unreliable, hence we use d_reclen.
    LIBC_NAMESPACE::memcpy(new_entry, entry, entry->d_reclen);

    if (!entries.push_back(new_entry)) {
      libc_errno = ENOMEM;
      break;
    }
  }

  LIBC_NAMESPACE::closedir(dir_fd);


  AllocChecker ac;
  struct dirent **result = new (ac) struct dirent*[entries.size()];
  if (!ac) {
    libc_errno = ENOMEM;
  }

  if (libc_errno != 0) {
    for (struct dirent *entry: entries) {
      delete entry;
    }
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
