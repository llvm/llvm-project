//===-- Linux implementation of the localtime function --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "localtime_utils.h"
#include "src/__support/File/file.h"
#include "src/time/linux/timezone.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace localtime_utils {

void release_file(ErrorOr<File *> error_or_file) {
  file_usage = 0;
  error_or_file.value()->close();
}

ErrorOr<File *> acquire_file(char *filename) {
  while (1) {
    if (file_usage == 0) {
      file_usage = 1;
      break;
    }
  }

  return LIBC_NAMESPACE::openfile(filename, "rb");
}

timezone::tzset *get_localtime(struct tm *tm) {
  (void)tm;
  return nullptr;
}

} // namespace localtime_utils
} // namespace LIBC_NAMESPACE_DECL
