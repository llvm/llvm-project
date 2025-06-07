/*===-- MemProfBinaryAccessData.inc - MemProf profiling runtime macros -*- C++
-*-======== *\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/
/*
 * This file defines the macros for memprof profiling data structures
 * for binary access.
 *
 * This file has two identical copies. The primary copy lives in LLVM and
 * the other one sits in compiler-rt/include/profile directory. To make changes
 * in this file, first modify the primary copy and copy it over to compiler-rt.
 * Testing of any change in this file can start only after the two copies are
 * synced up.
 *
\*===----------------------------------------------------------------------===*/

// A 64-bit magic number to uniquely identify the raw binary memprof binary
// access profile file.
#define MEMPROF_BINARY_ACCESS_RAW_MAGIC_64                                     \
  ((uint64_t)255 << 56 | (uint64_t)'f' << 48 | (uint64_t)'b' << 40 |           \
   (uint64_t)'m' << 32 | (uint64_t)'b' << 24 | (uint64_t)'i' << 16 |           \
   (uint64_t)'n' << 8 | (uint64_t)129)

// The version number of the raw binary format.
#define MEMPROF_BINARY_ACCESS_RAW_VERSION 1ULL

// A struct describing the header used for the raw binary memprof profile
// format.
PACKED(struct BinaryAccessHeader {
  uint64_t Magic;
  uint64_t Version;
  uint64_t TotalSize;
  uint64_t SegmentOffset;
  uint64_t NumSegments;
  uint64_t MemAddressOffset;
  uint64_t NumMemBlockAddresses;
});
