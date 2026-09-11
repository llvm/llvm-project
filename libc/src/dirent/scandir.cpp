//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of getpriority.
///
//===----------------------------------------------------------------------===//

#include "src/dirent/scandir.h"

#include "hdr/types/struct_dirent.h"
/*
#include "include/llvm-libc-types/__scandir_compare_t.h"
#include "include/llvm-libc-types/__scandir_filter_t.h"
*/
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/libc_errno.h"
#include "src/dirent/closedir.h"
#include "src/dirent/opendir.h"
#include "src/dirent/readdir.h"
#include "src/stdlib/qsort_util.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, scandir,
                   (const char *dir, struct dirent ***namelist,
                    __scandir_filter_t filter, __scandir_compare_t compare)) {
  (void)namelist;

  DIR *dir_fd = LIBC_NAMESPACE::opendir(dir);
  if (dir_fd == nullptr) {
    // errno set by opendir
    return -1;
  }

  struct dirent **namelist_local = {};
  (void)namelist_local;

  struct dirent *entry = nullptr;
  size_t count = 0;

  do {
    libc_errno = 0;
    entry = LIBC_NAMESPACE::readdir(dir_fd);
    if (libc_errno != 0) {
      // errno set by readdir
      return -1;
    }

    if (filter != nullptr && !filter(entry)) {
      continue;
    }

  } while (entry != nullptr);

  if (compare != nullptr) {
  }

  LIBC_NAMESPACE::closedir(dir_fd);

  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
