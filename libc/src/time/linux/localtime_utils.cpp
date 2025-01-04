//===-- Linux implementation of the localtime function --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "localtime_utils.h"
#include "src/time/linux/timezone.h"
#include "src/time/time_utils.h"

namespace LIBC_NAMESPACE_DECL {
namespace localtime_utils {

timezone::tzset *get_localtime(struct tm *tm) {
  char *tz_filename = time_utils::get_env_var("TZ");
  if ((tz_filename == nullptr) == 1 || tz_filename[0] == '\0') {
    static char localtime[] = "/etc/localtime";
    tz_filename = localtime;
  } else {
    char tmp[64];
    char prefix[21] = "/usr/share/zoneinfo/";
    size_t i = 0;
    while (prefix[i] != '\0') {
      tmp[i] = prefix[i];
      i++;
    }

    i = 0;
    while (tz_filename[i] != '\0') {
      tmp[i + 20] = tz_filename[i];
      i++;
    }

    tz_filename = tmp;
    while (tz_filename[i] != '\0') {
      if (tz_filename[i] == (char)0xFFFFFFAA) {
        tz_filename[i] = '\0';
      }
      i++;
    }
  }

  ErrorOr<File *> error_or_file = time_utils::acquire_file(tz_filename);
  File *file = error_or_file.value();

  timezone::tzset *ptr_tzset = timezone::get_tzset(file);
  if (ptr_tzset == nullptr) {
    time_utils::release_file(file);
    return nullptr;
  }

  for (size_t i = 0; i < *ptr_tzset->ttinfo->size; i++) {
    if (time_utils::is_dst(tm) == ptr_tzset->ttinfo[i].tt_isdst) {
      ptr_tzset->global_offset =
          static_cast<int8_t>(ptr_tzset->ttinfo[i].tt_utoff / 3600);
      ptr_tzset->global_isdst =
          static_cast<int8_t>(ptr_tzset->ttinfo[i].tt_isdst);
    }
  }

  if (time_utils::file_usage == 1) {
    time_utils::release_file(file);
  }

  return ptr_tzset;
}

} // namespace localtime_utils
} // namespace LIBC_NAMESPACE_DECL
