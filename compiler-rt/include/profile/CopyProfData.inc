//===-- CopyProfData.inc -----------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the common binary profile data format structures and magic
/// numbers shared between the CopyProf runtime and LLVM profile readers.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_DATA_INC
#define COPYPROF_DATA_INC

#include <stdint.h>
#include <string.h>

#ifdef _MSC_VER
#define PACKED(...) __pragma(pack(push, 1)) __VA_ARGS__ __pragma(pack(pop))
#else
#define PACKED(...) __VA_ARGS__ __attribute__((__packed__))
#endif

namespace llvm::copyprof {

// A 64-bit magic number to uniquely identify the raw binary CopyProf profile
// file.
inline constexpr uint64_t RawMagic64 =
    (uint64_t{255} << 56 | uint64_t{'c'} << 48 | uint64_t{'s'} << 40 |
     uint64_t{'a'} << 32 | uint64_t{'n'} << 24 | uint64_t{'p'} << 16 |
     uint64_t{'r'} << 8 | uint64_t{129});
inline constexpr uint64_t RawVersion = 1;
inline constexpr uint64_t BuildIdMaxSize = 32;

// The header used in the CopyProf report file format.
PACKED(struct Header {
  // An arbitrary 64-bit constant. Must match `RawMagic64` otherwise the
  // file will be rejected.
  uint64_t Magic;
  // The version of this report file.
  uint64_t Version;
  // The byte offset from the start of the file to the report section.
  uint64_t ReportOffset;
  // The number of `Report` entries in the report section.
  uint64_t NumReports;
  // The byte offset from the start of the file to the segment section.
  uint64_t SegmentOffset;
  // The byte offset from the start of the file to the stack trace section.
  uint64_t StackOffset;
});

// Describes an executable memory segment loaded at runtime. Used during
// symbolization to map runtime virtual addresses back to the binary.
PACKED(struct SegmentEntry {
  // The runtime start address of the executable segment.
  uint64_t Start;
  // The runtime end address (exclusive) of the executable segment.
  uint64_t End;
  // The base address at which the containing module was loaded.
  uint64_t Offset;
  // The number of valid bytes in `BuildId`.
  uint64_t BuildIdSize;
  // The build ID of the module, used to match segments to binaries.
  uint8_t BuildId[BuildIdMaxSize] = {0};

  SegmentEntry() = default;

  SegmentEntry(const SegmentEntry &S) {
    Start = S.Start;
    End = S.End;
    Offset = S.Offset;
    BuildIdSize = S.BuildIdSize;
    memcpy(BuildId, S.BuildId, S.BuildIdSize);
  }

  SegmentEntry &operator=(const SegmentEntry &S) {
    Start = S.Start;
    End = S.End;
    Offset = S.Offset;
    BuildIdSize = S.BuildIdSize;
    memcpy(BuildId, S.BuildId, S.BuildIdSize);
    return *this;
  }

  bool operator==(const SegmentEntry &S) const {
    return Start == S.Start && End == S.End && Offset == S.Offset &&
           BuildIdSize == S.BuildIdSize &&
           memcmp(BuildId, S.BuildId, S.BuildIdSize) == 0;
  }
});

// A single report of an unnecessary copy detected at runtime.
PACKED(struct Report {
  // An ID into the stack trace section identifying the call stack where the
  // unnecessary copy was destroyed.
  uint32_t stack_trace_id;
  // The size in bytes of the unnecessary copy.
  uint64_t copy_num_bytes;
});

} // namespace llvm::copyprof

#endif // COPYPROF_DATA_INC
