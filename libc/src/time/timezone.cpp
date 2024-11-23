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

#include "src/time/time_utils.h"
#include "src/__support/common.h"
#include "src/time/timezone.h"

namespace LIBC_NAMESPACE_DECL {
namespace timezone {

using LIBC_NAMESPACE::time_utils::TimeConstants;

void rev_str(char *str) {
    int start = 0;
    int end = 0;

    while (str[end] != '\0') {
        end++;
    }
    end--;

    while (start < end) {
        str[start] = str[start] ^ str[end];
        str[end] = str[start] ^ str[end];
        str[start] = str[start] ^ str[end];

        start++;
        end--;
    }
}

int get_timezone_offset(char *timezone) {
  (void)timezone;

  unsigned char hdr[TIMEZONE_HDR_SIZE + 1];

  int32_t magic;
  unsigned char version;
  __int128_t reserved;
  int32_t tzh_ttisutcnt;
  int32_t tzh_ttisstdcnt;
  int32_t tzh_leapcnt;
  int32_t tzh_timecnt;
  int32_t tzh_typecnt;
  int32_t tzh_charcnt;

  int fd;
  size_t bytes;

  fd = open("/etc/localtime", O_RDONLY);
  if (fd < 0) {
    close(fd);
    return 0;
  }

  bytes = read(fd, hdr, sizeof(hdr));
  if (bytes != sizeof(hdr)) {
    close(fd);
    return 0;
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
  tzh_ttisutcnt = (hdr[21] << 24) | (hdr[22] << 16) | (hdr[23] << 8) | hdr[24];
  tzh_ttisstdcnt = (hdr[25] << 24) | (hdr[26] << 16) | (hdr[27] << 8) | hdr[28];
  tzh_leapcnt = (hdr[29] << 24) | (hdr[30] << 16) | (hdr[31] << 8) | hdr[32];
  tzh_timecnt = (hdr[33] << 24) | (hdr[34] << 16) | (hdr[35] << 8) | hdr[36];
  tzh_typecnt = (hdr[37] << 24) | (hdr[38] << 16) | (hdr[39] << 8) | hdr[40];
  tzh_charcnt = (hdr[41] << 24) | (hdr[42] << 16) | (hdr[43] << 8) | hdr[44];
  (void)tzh_ttisutcnt;
  (void)tzh_ttisstdcnt;
  (void)tzh_leapcnt;
  (void)tzh_typecnt;
  (void)tzh_charcnt;

  if (magic != 0x545A6966) {
    return 0;
  }

  if (version != 0x32 && version != 0x33 && version != 0x34) {
    return 0;
  }

  // according to `tzfile`, 15 bytes should be 0
  if (reserved != 0) {
    return 0;
  }

  for (i = 0; i < (size_t)tzh_timecnt; i++) {
    uint8_t buf[4];
    bytes = read(fd, buf, 4);
    if (bytes != 4) {
      continue;
    }

    int32_t transition = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
    transition = ((transition & 0xFF000000) >> 24) |
        ((transition & 0x00FF0000) >> 8) |
        ((transition & 0x0000FF00) << 8) |
        ((transition & 0x000000FF) << 24);
    printf("transition %d:   %d\n", i, transition);
  }

  close(fd);

  return 0;
}

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL
