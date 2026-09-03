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
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, scandir,
                   (const char *dir, struct dirent ***namelist,
                    int (*sel)(const struct dirent *),
                    int (*compar)(const struct dirent **,
                                  const struct dirent **))) {
  (void)dir;
  (void)namelist;
  (void)sel;
  (void)compar;
  return -1;
}

} // namespace LIBC_NAMESPACE_DECL
