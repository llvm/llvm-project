//===-- Implementation of timezone functions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_TIMEZONE_H
#define LLVM_LIBC_SRC_TIME_TIMEZONE_H

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "stdint.h"

namespace LIBC_NAMESPACE_DECL {
namespace timezone {

// accoring to `tzfile`, timezone header if always 44 bytes
#define TIMEZONE_HDR_SIZE 44

typedef struct {
    int64_t *tt_utoff;
    uint8_t *tt_isdst;
    uint8_t *tt_desigidx;

    // additional fields
    int64_t *offsets;
} ttinfo;

typedef struct {
    uint64_t tzh_ttisutcnt;
    uint64_t tzh_ttisstdcnt;
    uint64_t tzh_leapcnt;
    uint64_t tzh_timecnt;
    uint64_t tzh_typecnt;
    uint64_t tzh_charcnt;
    ttinfo *ttinfo;

    // additional fields
    int64_t *tzh_timecnt_transitions;
    int64_t *tzh_timecnt_indices;
    size_t tzh_timecnt_number_transitions;
    unsigned char *tz;
} tzset;

tzset *get_tzset(int fd);

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_TIMEZONE_H
