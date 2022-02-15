#ifndef MEMPROF_DATA_INC
#define MEMPROF_DATA_INC
/*===-- MemProfData.inc - MemProf profiling runtime structures -*- C++ -*-=== *\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/
/*
 * This is the main file that defines all the data structure, signature,
 * constant literals that are shared across profiling runtime library,
 * and host tools (reader/writer).
 *
 * This file has two identical copies. The primary copy lives in LLVM and
 * the other one sits in compiler-rt/include/profile directory. To make changes
 * in this file, first modify the primary copy and copy it over to compiler-rt.
 * Testing of any change in this file can start only after the two copies are
 * synced up.
 *
\*===----------------------------------------------------------------------===*/

#ifdef _MSC_VER
#define PACKED(...) __pragma(pack(push,1)) __VA_ARGS__ __pragma(pack(pop))
#else
#define PACKED(...) __VA_ARGS__ __attribute__((__packed__))
#endif

// A 64-bit magic number to uniquely identify the raw binary memprof profile file.
#define MEMPROF_RAW_MAGIC_64                                                                        \
  ((uint64_t)255 << 56 | (uint64_t)'m' << 48 | (uint64_t)'p' << 40 | (uint64_t)'r' << 32 |          \
   (uint64_t)'o' << 24 | (uint64_t)'f' << 16 | (uint64_t)'r' << 8 | (uint64_t)129)

// The version number of the raw binary format.
#define MEMPROF_RAW_VERSION 1ULL

namespace llvm {
namespace memprof {
// A struct describing the header used for the raw binary memprof profile format.
PACKED(struct Header {
  uint64_t Magic;
  uint64_t Version;
  uint64_t TotalSize;
  uint64_t SegmentOffset;
  uint64_t MIBOffset;
  uint64_t StackOffset;
});


// A struct describing the information necessary to describe a /proc/maps
// segment entry for a particular binary/library identified by its build id.
PACKED(struct SegmentEntry {
  uint64_t Start;
  uint64_t End;
  uint64_t Offset;
  // This field is unused until sanitizer procmaps support for build ids for
  // Linux-Elf is implemented.
  uint8_t BuildId[32] = {0};

  SegmentEntry(uint64_t S, uint64_t E, uint64_t O) :
    Start(S), End(E), Offset(O) {}

  SegmentEntry(const SegmentEntry& S) {
    Start = S.Start;
    End = S.End;
    Offset = S.Offset;
  }

  SegmentEntry& operator=(const SegmentEntry& S) {
    Start = S.Start;
    End = S.End;
    Offset = S.Offset;
    return *this;
  }

  bool operator==(const SegmentEntry& S) const {
    return Start == S.Start &&
           End == S.End &&
           Offset == S.Offset;
  }
});

// A struct representing the heap allocation characteristics of a particular
// runtime context. This struct is shared between the compiler-rt runtime and
// the raw profile reader. The indexed format uses a separate, self-describing
// backwards compatible format.
PACKED(struct MemInfoBlock {
  uint32_t alloc_count;
  uint64_t total_access_count, min_access_count, max_access_count;
  uint64_t total_size;
  uint32_t min_size, max_size;
  uint32_t alloc_timestamp, dealloc_timestamp;
  uint64_t total_lifetime;
  uint32_t min_lifetime, max_lifetime;
  uint32_t alloc_cpu_id, dealloc_cpu_id;
  uint32_t num_migrated_cpu;

  // Only compared to prior deallocated object currently.
  uint32_t num_lifetime_overlaps;
  uint32_t num_same_alloc_cpu;
  uint32_t num_same_dealloc_cpu;

  uint64_t data_type_id; // TODO: hash of type name

  MemInfoBlock() : alloc_count(0) {}

  MemInfoBlock(uint32_t size, uint64_t access_count, uint32_t alloc_timestamp,
               uint32_t dealloc_timestamp, uint32_t alloc_cpu, uint32_t dealloc_cpu)
      : alloc_count(1), total_access_count(access_count),
        min_access_count(access_count), max_access_count(access_count),
        total_size(size), min_size(size), max_size(size),
        alloc_timestamp(alloc_timestamp), dealloc_timestamp(dealloc_timestamp),
        total_lifetime(dealloc_timestamp - alloc_timestamp),
        min_lifetime(total_lifetime), max_lifetime(total_lifetime),
        alloc_cpu_id(alloc_cpu), dealloc_cpu_id(dealloc_cpu),
        num_lifetime_overlaps(0), num_same_alloc_cpu(0),
        num_same_dealloc_cpu(0) {
    num_migrated_cpu = alloc_cpu_id != dealloc_cpu_id;
  }

  void Merge(const MemInfoBlock &newMIB) {
    alloc_count += newMIB.alloc_count;

    total_access_count += newMIB.total_access_count;
    min_access_count = newMIB.min_access_count < min_access_count ? newMIB.min_access_count : min_access_count;
    max_access_count = newMIB.max_access_count < max_access_count ? newMIB.max_access_count : max_access_count;

    total_size += newMIB.total_size;
    min_size = newMIB.min_size < min_size ? newMIB.min_size : min_size;
    max_size = newMIB.max_size < max_size ? newMIB.max_size : max_size;

    total_lifetime += newMIB.total_lifetime;
    min_lifetime = newMIB.min_lifetime < min_lifetime ? newMIB.min_lifetime : min_lifetime;
    max_lifetime = newMIB.max_lifetime > max_lifetime ? newMIB.max_lifetime : max_lifetime;

    // We know newMIB was deallocated later, so just need to check if it was
    // allocated before last one deallocated.
    num_lifetime_overlaps += newMIB.alloc_timestamp < dealloc_timestamp;
    alloc_timestamp = newMIB.alloc_timestamp;
    dealloc_timestamp = newMIB.dealloc_timestamp;

    num_same_alloc_cpu += alloc_cpu_id == newMIB.alloc_cpu_id;
    num_same_dealloc_cpu += dealloc_cpu_id == newMIB.dealloc_cpu_id;
    alloc_cpu_id = newMIB.alloc_cpu_id;
    dealloc_cpu_id = newMIB.dealloc_cpu_id;
  }
});

} // namespace memprof
} // namespace llvm

#endif
