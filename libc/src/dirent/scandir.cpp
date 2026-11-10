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

#include "hdr/types/struct_dirent.h"
#include "src/__support/File/dir.h"
#include "src/__support/common.h"
#include "src/__support/libc_errno.h"
#include "src/__support/macros/null_check.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, scandir,
                   (const char *dir, struct dirent ***namelist,
                    int (*filter)(const struct dirent *),
                    int (*compare)(const struct dirent **,
                                   const struct dirent **))) {
  LIBC_CRASH_ON_NULLPTR(dir);
  LIBC_CRASH_ON_NULLPTR(namelist);

  *namelist = nullptr;

  const auto res = Dir::scan(dir, filter, compare);
  if (LIBC_UNLIKELY(!res)) {
    libc_errno = res.error();
    return -1;
  }

  const DirScanResult &scan_res = res.value();
  *namelist = scan_res.entries;
  return static_cast<int>(scan_res.count);
}

} // namespace LIBC_NAMESPACE_DECL
