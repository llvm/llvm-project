//===-- Implementation of timezone functions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h> // TODO: Remove all printf functions
#include <sys/types.h>

#include "src/__support/common.h"
#include "src/time/timezone.h"

namespace LIBC_NAMESPACE_DECL {
namespace timezone {

tzset *get_timezone_offset(char *timezone) {
  static ttinfo ttinfo;
  static tzset result;

  unsigned char hdr[TIMEZONE_HDR_SIZE * 10];

  int64_t magic;
  unsigned char version;
  __int128_t reserved;
  uint32_t tzh_ttisutcnt;
  uint32_t tzh_ttisstdcnt;
  uint32_t tzh_leapcnt;
  uint32_t tzh_timecnt;
  uint32_t tzh_typecnt;
  uint32_t tzh_charcnt;

  int fd;
  size_t bytes;

  fd = open("/etc/localtime", O_RDONLY);
  if (fd < 0) {
    close(fd);
    return nullptr;
  }

  bytes = read(fd, hdr, sizeof(hdr));
  // TODO: Remove the number of bytes to check
  if (bytes != 379) {
      close(fd);
      return nullptr;
  }

  size_t i;
  __uint128_t tmp;

  // these locations in timezone files are defined in documentation
  // for `tzfile`
  magic = (hdr[0] << 24) | (hdr[1] << 16) | (hdr[2] << 8) | hdr[3];
  version = hdr[4];
  for (i = 5; i < 21; i++) {
    tmp = (tmp << 8) | hdr[i];
  }
  reserved = tmp;
  tzh_ttisutcnt = (hdr[20] << 24) | (hdr[21] << 16) | (hdr[22] << 8) | hdr[23];
  tzh_ttisstdcnt = (hdr[24] << 24) | (hdr[25] << 16) | (hdr[26] << 8) | hdr[27];
  tzh_leapcnt = (hdr[28] << 24) | (hdr[29] << 16) | (hdr[30] << 8) | hdr[31];
  tzh_timecnt = (hdr[32] << 24) | (hdr[33] << 16) | (hdr[34] << 8) | hdr[35];
  tzh_typecnt = (hdr[36] << 24) | (hdr[37] << 16) | (hdr[38] << 8) | hdr[39];
  tzh_charcnt = (hdr[40] << 24) | (hdr[41] << 16) | (hdr[42] << 8) | hdr[43];

  result.tzh_ttisutcnt = tzh_ttisutcnt;
  result.tzh_ttisstdcnt = tzh_ttisstdcnt;
  result.tzh_leapcnt = tzh_leapcnt;
  result.tzh_timecnt = tzh_timecnt;
  result.tzh_typecnt = tzh_typecnt;
  result.tzh_charcnt = tzh_charcnt;

  if (magic != 0x545A6966) {
    return nullptr;
  }

  if (version != 0x32 && version != 0x33 && version != 0x34) {
    return nullptr;
  }

  // according to `tzfile`, 15 bytes should be 0
  if (reserved != 0) {
    return nullptr;
  }

  close(fd);

  return &result;
}

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL
