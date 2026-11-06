//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Constants shared by the host and device ConcurrencySanitizer runtimes.
///
//===----------------------------------------------------------------------===//

#ifndef CSAN_DEFS_H
#define CSAN_DEFS_H

enum {
  CSAN_ACCESS_ATOMIC = 1u << 0,
  CSAN_ACCESS_COMPOUND = 1u << 1,
  CSAN_ACCESS_WRITE = 1u << 2
};

enum {
  CSAN_RACE_DATA = 0,
  CSAN_RACE_UNKNOWN_ORIGIN = 1,
  CSAN_RACE_INTRA_WAVE = 2
};

// The size of the device watchpoint table used by the host and device runtime.
static constexpr unsigned long CSAN_WATCHPOINT_TABLE_BYTES =
    2ul * 1024ul * 1024ul;
static constexpr unsigned long CSAN_WATCHPOINT_TABLE_ENTRIES =
    CSAN_WATCHPOINT_TABLE_BYTES / sizeof(unsigned long long);

#endif // CSAN_DEFS_H
