//===-- Implementation of timezone functions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include "src/__support/common.h"
#include "src/time/time_utils.h"
#include "src/time/timezone.h"

namespace LIBC_NAMESPACE_DECL {
namespace timezone {

tzset *get_tzset(File *file) {
  unsigned char hdr[TIMEZONE_HDR_SIZE + 4096];
  int64_t magic;
  unsigned char version;
  __int128_t reserved;
  uint32_t tzh_ttisutcnt;
  uint32_t tzh_ttisstdcnt;
  uint32_t tzh_leapcnt;
  uint32_t tzh_timecnt;
  uint32_t tzh_typecnt;
  uint32_t tzh_charcnt;
  __uint128_t tmp;
  size_t i;

  file->read(hdr, TIMEZONE_HDR_SIZE + 4096);

  // these locations are defined in documentation
  // for `tzfile` and should be 44 bytes
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

  static tzset result;

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

  product = (tzh_timecnt * 5) + (tzh_typecnt * 6) + (tzh_leapcnt * 8) +
            tzh_charcnt + tzh_ttisstdcnt + tzh_ttisutcnt + TIMEZONE_HDR_SIZE;

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

  int64_t tzh_timecnt_transitions[chunk];
  int64_t *ptr_tzh_timecnt_transitions;

  ptr_tzh_timecnt_transitions = tzh_timecnt_transitions;
  for (i = 0; i < chunk; ++i) {
    *(ptr_tzh_timecnt_transitions + i) =
        (static_cast<int64_t>(hdr[start + i * 8]) << 56) |
        (static_cast<int64_t>(hdr[start + i * 8 + 1]) << 48) |
        (static_cast<int64_t>(hdr[start + i * 8 + 2]) << 40) |
        (static_cast<int64_t>(hdr[start + i * 8 + 3]) << 32) |
        (static_cast<int64_t>(hdr[start + i * 8 + 4]) << 24) |
        (static_cast<int64_t>(hdr[start + i * 8 + 5]) << 16) |
        (static_cast<int64_t>(hdr[start + i * 8 + 6]) << 8) |
        static_cast<int64_t>(hdr[start + i * 8 + 7]);
  }
  result.tzh_timecnt_transitions = ptr_tzh_timecnt_transitions;
  result.tzh_timecnt_number_transitions = chunk + 1;

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

  int64_t tz[tzh_charcnt_end - tzh_leapcnt_end - 1];
  int64_t *ptr_tz;

  ptr_tz = tz;
  result.tz = ptr_tz;
  j = 0;
  for (i = tzh_leapcnt_end; i < static_cast<size_t>(tzh_charcnt_end - 1); ++i) {
    if (i == static_cast<size_t>(tzh_charcnt_end - 1)) {
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

  chunk = ((tzh_typecnt_end - tzh_timecnt_end) / 6);
  ttinfo ttinfo[chunk];

  size_t index = 0;
  for (size_t i = tzh_timecnt_end; i < static_cast<size_t>(tzh_typecnt_end);
       i += 6) {
    int32_t tt_utoff = static_cast<int32_t>(hdr[i] << 24) |
                       static_cast<int32_t>(hdr[i + 1] << 16) |
                       static_cast<int32_t>(hdr[i + 2] << 8) |
                       static_cast<int32_t>(hdr[i + 3]);
    uint8_t tt_isdst = hdr[i + 4];
    size_t tt_desigidx = hdr[i + 5];

    size_t k = 0;
    for (size_t j = 0; j < tt_desigidx; j++) {
      if (tz[j] == ';') {
        k++;
      }
    }

    ttinfo[index].tt_utoff = tt_utoff;
    ttinfo[index].tt_isdst = tt_isdst;
    ttinfo[index].tt_desigidx = static_cast<int8_t>(k);

    ttinfo[index].size = &chunk;

    index++;
  }

  result.ttinfo = ttinfo;

  return &result;
}

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL
