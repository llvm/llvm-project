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

tzset *get_tzset(int fd) {
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

  size_t bytes;

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

  int64_t product;

  product = (tzh_timecnt * 5)
      + (tzh_typecnt * 6)
      + (tzh_leapcnt * 8)
      + tzh_charcnt
      + tzh_ttisstdcnt
      + tzh_ttisutcnt
      + TIMEZONE_HDR_SIZE;

  int64_t tzh_timecnt_length;
  int64_t tzh_typecnt_length;
  int64_t tzh_leapcnt_length;
  int64_t tzh_charcnt_length;
  int64_t tzh_timecnt_end;
  int64_t tzh_typecnt_end;
  int64_t tzh_leapcnt_end;
  int64_t tzh_charcnt_end;

  tzh_timecnt_length = tzh_timecnt * 9;
  tzh_typecnt_length = tzh_typecnt * 6;
  tzh_leapcnt_length = tzh_leapcnt * 12;
  tzh_charcnt_length = tzh_charcnt;
  tzh_timecnt_end = TIMEZONE_HDR_SIZE + product + tzh_timecnt_length;
  tzh_typecnt_end = tzh_timecnt_end + tzh_typecnt_length;
  tzh_leapcnt_end = tzh_typecnt_end + tzh_leapcnt_length;
  tzh_charcnt_end = tzh_leapcnt_end + tzh_charcnt_length;

  size_t start;
  size_t end;
  size_t chunk;

  start = TIMEZONE_HDR_SIZE + product;
  end = (TIMEZONE_HDR_SIZE + product + (tzh_timecnt * 8));
  chunk = (end - start) / 8;

  int64_t tzh_timecnt_transitions[chunk + 1];
  int64_t *ptr_tzh_timecnt_transitions;

  ptr_tzh_timecnt_transitions = tzh_timecnt_transitions;
  for (i = 0; i < chunk; ++i) {
    *(ptr_tzh_timecnt_transitions + i) = ((int64_t)hdr[start + i * 8] << 56) |
                          ((int64_t)hdr[start + i * 8 + 1] << 48) |
                          ((int64_t)hdr[start + i * 8 + 2] << 40) |
                          ((int64_t)hdr[start + i * 8 + 3] << 32) |
                          ((int64_t)hdr[start + i * 8 + 4] << 24) |
                          ((int64_t)hdr[start + i * 8 + 5] << 16) |
                          ((int64_t)hdr[start + i * 8 + 6] << 8) |
                          (int64_t)hdr[start + i * 8 + 7];
  }
  result.tzh_timecnt_transitions = ptr_tzh_timecnt_transitions;

  start = TIMEZONE_HDR_SIZE + product + tzh_timecnt * 8;
  end = tzh_timecnt_end;

  int64_t tzh_timecnt_indices[end - start];
  int64_t *ptr_tzh_timecnt_indices;
  size_t j;

  ptr_tzh_timecnt_indices = tzh_timecnt_indices;
  j = 0;
  for (i = start; i < end; ++i) {
      tzh_timecnt_indices[j] = hdr[i];
      j += 1;
  }
  result.tzh_timecnt_indices = ptr_tzh_timecnt_indices;

  unsigned char tz[tzh_charcnt_end - tzh_leapcnt_end];
  unsigned char *ptr_tz;

  ptr_tz = tz;
  j = 0;
  for (i = tzh_leapcnt_end; i < (size_t)tzh_charcnt_end + 1; ++i) {
      if (i == (size_t)tzh_charcnt_end - 1) {
          tz[j] = '\0';
          break;
      }

      if (hdr[i] == '\0') {
          tz[j] = 0x3B;
          j += 1;
          continue;
      }

      tz[j] = hdr[i];

      j += 1;
  }
  result.tz = ptr_tz;

  int64_t offsets[6];
  int64_t *ptr_offsets;
  size_t index;

  int64_t tt_utoff[6];
  uint8_t tt_isdst[6];
  uint8_t tt_desigidx[6];

  int64_t *ptr_tt_utoff;
  uint8_t *ptr_tt_isdst;
  uint8_t *ptr_tt_desigidx;

  ptr_offsets = offsets;
  ptr_tt_utoff = tt_utoff;
  ptr_tt_isdst = tt_isdst;
  ptr_tt_desigidx = tt_desigidx;

  index = 0;
  for (size_t i = tzh_timecnt_end; i < (size_t)tzh_typecnt_end; i += 6) {
      unsigned char *tmp;

      tmp = &hdr[i];
      *(ptr_offsets + index) = tmp[5];

      *(ptr_tt_utoff + index) = tmp[0] << 24 | tmp[1] << 16 | tmp[2] << 8 | tmp[3];
      *(ptr_tt_isdst + index) = tmp[4];
      *(ptr_tt_desigidx + index) = (uint8_t)index;

      index += 1;
  }

  ttinfo.offsets = ptr_offsets;
  ttinfo.tt_utoff = ptr_tt_utoff;
  ttinfo.tt_isdst = ptr_tt_isdst;
  ttinfo.tt_desigidx = ptr_tt_desigidx;

  result.ttinfo = &ttinfo;

  close(fd);

  return &result;
}

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL
