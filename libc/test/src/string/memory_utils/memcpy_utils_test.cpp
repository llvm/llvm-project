//===-- Unittests for memory_utils ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memory_utils/memcpy_utils.h"
#include "utils/CPP/Array.h"
#include "utils/UnitTest/Test.h"

#include <assert.h>
#include <stdint.h> // uintptr_t

#ifndef LLVM_LIBC_MEMCPY_MONITOR
#error LLVM_LIBC_MEMCPY_MONITOR must be defined for this test.
#endif

namespace __llvm_libc {

struct Buffer {
  static constexpr size_t kMaxBuffer = 1024;
  char buffer[kMaxBuffer + 1];
  size_t last = 0;

  void Clear() {
    last = 0;
    for (size_t i = 0; i < kMaxBuffer; ++i)
      buffer[i] = '0';
    buffer[kMaxBuffer] = '\0';
  }

  void Increment(const void *ptr) {
    const auto offset = reinterpret_cast<uintptr_t>(ptr);
    assert(offset < kMaxBuffer);
    ++buffer[offset];
    if (offset > last)
      last = offset;
  }

  char *Finish() {
    assert(last < kMaxBuffer);
    buffer[last + 1] = '\0';
    return buffer;
  }
};

struct Trace {
  Buffer read;
  Buffer write;

  void Add(char *__restrict dst, const char *__restrict src, size_t count) {
    for (size_t i = 0; i < count; ++i)
      read.Increment(src + i);
    for (size_t i = 0; i < count; ++i)
      write.Increment(dst + i);
  }

  void Clear() {
    read.Clear();
    write.Clear();
  }

  char *Read() { return read.Finish(); }
  char *Write() { return write.Finish(); }
};

static Trace &GetTrace() {
  static thread_local Trace events;
  return events;
}

extern "C" void LLVM_LIBC_MEMCPY_MONITOR(char *__restrict dst,
                                         const char *__restrict src,
                                         size_t count) {
  GetTrace().Add(dst, src, count);
}

char *I(uintptr_t offset) { return reinterpret_cast<char *>(offset); }

TEST(MemcpyUtilsTest, CopyTrivial) {
  auto &trace = GetTrace();

  trace.Clear();
  CopyBlock<1>(I(0), I(0));
  EXPECT_STREQ(trace.Write(), "1");
  EXPECT_STREQ(trace.Read(), "1");

  trace.Clear();
  CopyBlock<2>(I(0), I(0));
  EXPECT_STREQ(trace.Write(), "11");
  EXPECT_STREQ(trace.Read(), "11");

  trace.Clear();
  CopyBlock<4>(I(0), I(0));
  EXPECT_STREQ(trace.Write(), "1111");
  EXPECT_STREQ(trace.Read(), "1111");

  trace.Clear();
  CopyBlock<8>(I(0), I(0));
  EXPECT_STREQ(trace.Write(), "11111111");
  EXPECT_STREQ(trace.Read(), "11111111");

  trace.Clear();
  CopyBlock<16>(I(0), I(0));
  EXPECT_STREQ(trace.Write(), "1111111111111111");
  EXPECT_STREQ(trace.Read(), "1111111111111111");

  trace.Clear();
  CopyBlock<32>(I(0), I(0));
  EXPECT_STREQ(trace.Write(), "11111111111111111111111111111111");
  EXPECT_STREQ(trace.Read(), "11111111111111111111111111111111");

  trace.Clear();
  CopyBlock<64>(I(0), I(0));
  EXPECT_STREQ(
      trace.Write(),
      "1111111111111111111111111111111111111111111111111111111111111111");
  EXPECT_STREQ(
      trace.Read(),
      "1111111111111111111111111111111111111111111111111111111111111111");
}

TEST(MemcpyUtilsTest, CopyOffset) {
  auto &trace = GetTrace();

  trace.Clear();
  CopyBlock<1>(I(3), I(1));
  EXPECT_STREQ(trace.Write(), "0001");
  EXPECT_STREQ(trace.Read(), "01");

  trace.Clear();
  CopyBlock<1>(I(2), I(1));
  EXPECT_STREQ(trace.Write(), "001");
  EXPECT_STREQ(trace.Read(), "01");
}

TEST(MemcpyUtilsTest, CopyBlockOverlap) {
  auto &trace = GetTrace();

  trace.Clear();
  CopyBlockOverlap<2>(I(0), I(0), 2);
  EXPECT_STREQ(trace.Write(), "22");
  EXPECT_STREQ(trace.Read(), "22");

  trace.Clear();
  CopyBlockOverlap<2>(I(0), I(0), 3);
  EXPECT_STREQ(trace.Write(), "121");
  EXPECT_STREQ(trace.Read(), "121");

  trace.Clear();
  CopyBlockOverlap<2>(I(0), I(0), 4);
  EXPECT_STREQ(trace.Write(), "1111");
  EXPECT_STREQ(trace.Read(), "1111");

  trace.Clear();
  CopyBlockOverlap<4>(I(2), I(1), 7);
  EXPECT_STREQ(trace.Write(), "001112111");
  EXPECT_STREQ(trace.Read(), "01112111");
}

TEST(MemcpyUtilsTest, CopyAlignedBlocks) {
  auto &trace = GetTrace();
  // Destination is aligned already.
  //   "1111000000000"
  // + "0000111100000"
  // + "0000000011110"
  // + "0000000001111"
  // = "1111111112221"
  trace.Clear();
  CopyAlignedBlocks<4>(I(0), I(0), 13);
  EXPECT_STREQ(trace.Write(), "1111111112221");
  EXPECT_STREQ(trace.Read(), "1111111112221");

  // Misaligned destination
  //   "01111000000000"
  // + "00001111000000"
  // + "00000000111100"
  // + "00000000001111"
  // = "01112111112211"
  trace.Clear();
  CopyAlignedBlocks<4>(I(1), I(0), 13);
  EXPECT_STREQ(trace.Write(), "01112111112211");
  EXPECT_STREQ(trace.Read(), "1112111112211");
}

TEST(MemcpyUtilsTest, MaxReloads) {
  auto &trace = GetTrace();
  for (size_t alignment = 0; alignment < 32; ++alignment) {
    for (size_t count = 64; count < 768; ++count) {
      trace.Clear();
      // We should never reload more than twice when copying from count = 2x32.
      CopyAlignedBlocks<32>(I(alignment), I(0), count);
      const char *const written = trace.Write();
      // First bytes are untouched.
      for (size_t i = 0; i < alignment; ++i)
        EXPECT_EQ(written[i], '0');
      // Next bytes are loaded once or twice but no more.
      for (size_t i = alignment; i < count; ++i) {
        EXPECT_GE(written[i], '1');
        EXPECT_LE(written[i], '2');
      }
    }
  }
}

} // namespace __llvm_libc
