//===-- Unittests for memory_utils ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define LLVM_LIBC_UNITTEST_OBSERVE 1

#include "src/string/memory_utils/elements.h"
#include "utils/CPP/Array.h"
#include "utils/CPP/ArrayRef.h"
#include "utils/UnitTest/Test.h"

#include <stdio.h>
#include <string.h>

namespace __llvm_libc {

static constexpr const size_t kMaxBuffer = 32;

struct BufferAccess : cpp::Array<char, kMaxBuffer + 1> {
  BufferAccess() { Reset(); }
  void Reset() {
    for (auto &value : *this)
      value = '0';
    this->operator[](kMaxBuffer) = '\0';
  }
  void Touch(ptrdiff_t offset, size_t size) {
    if (offset < 0)
      return;
    for (size_t i = 0; i < size; ++i)
      ++(*this)[offset + i];
  }
  operator const char *() const { return this->data(); }
};

struct Buffer {
  ptrdiff_t Offset(const char *ptr) const {
    const bool contained = ptr >= data.begin() && ptr < data.end();
    return contained ? ptr - data.begin() : -1;
  }
  void Reset() {
    reads.Reset();
    writes.Reset();
  }
  cpp::Array<char, kMaxBuffer> data;
  BufferAccess __attribute__((aligned(64))) reads;
  BufferAccess __attribute__((aligned(64))) writes;
};

struct MemoryAccessObserver {
  void ObserveRead(const char *ptr, size_t size) {
    Buffer1.reads.Touch(Buffer1.Offset(ptr), size);
    Buffer2.reads.Touch(Buffer2.Offset(ptr), size);
  }

  void ObserveWrite(const char *ptr, size_t size) {
    Buffer1.writes.Touch(Buffer1.Offset(ptr), size);
    Buffer2.writes.Touch(Buffer2.Offset(ptr), size);
  }

  void Reset() {
    Buffer1.Reset();
    Buffer2.Reset();
  }

  Buffer Buffer1;
  Buffer Buffer2;
};

MemoryAccessObserver Observer;

template <size_t Size> struct TestingElement {
  static constexpr size_t kSize = Size;

  static void Copy(char *__restrict dst, const char *__restrict src) {
    Observer.ObserveRead(src, kSize);
    Observer.ObserveWrite(dst, kSize);
  }

  static bool Equals(const char *lhs, const char *rhs) {
    Observer.ObserveRead(lhs, kSize);
    Observer.ObserveRead(rhs, kSize);
    return true;
  }

  static int ThreeWayCompare(const char *lhs, const char *rhs) {
    Observer.ObserveRead(lhs, kSize);
    Observer.ObserveRead(rhs, kSize);
    return 0;
  }

  static void SplatSet(char *dst, const unsigned char value) {
    Observer.ObserveWrite(dst, kSize);
  }
};

using Types = testing::TypeList<
    TestingElement<1>,                                               // 1 Byte
    TestingElement<2>,                                               // 2 Bytes
    TestingElement<4>,                                               // 4 Bytes
    Repeated<TestingElement<2>, 3>,                                  // 6 Bytes
    Chained<TestingElement<4>, TestingElement<2>, TestingElement<1>> // 7 Bytes
    >;

struct LlvmLibcTestAccessBase : public testing::Test {

  template <typename HigherOrder, size_t Size, size_t Offset = 0>
  void checkOperations(const BufferAccess &expected) {
    static const BufferAccess untouched;

    Observer.Reset();
    HigherOrder::Copy(dst_ptr() + Offset, src_ptr() + Offset, Size);
    ASSERT_STREQ(src().writes, untouched);
    ASSERT_STREQ(dst().reads, untouched);
    ASSERT_STREQ(src().reads, expected);
    ASSERT_STREQ(dst().writes, expected);
    Observer.Reset();
    HigherOrder::Equals(lhs_ptr() + Offset, rhs_ptr() + Offset, Size);
    ASSERT_STREQ(lhs().writes, untouched);
    ASSERT_STREQ(rhs().writes, untouched);
    ASSERT_STREQ(lhs().reads, expected);
    ASSERT_STREQ(rhs().reads, expected);
    Observer.Reset();
    HigherOrder::ThreeWayCompare(lhs_ptr() + Offset, rhs_ptr() + Offset, Size);
    ASSERT_STREQ(lhs().writes, untouched);
    ASSERT_STREQ(rhs().writes, untouched);
    ASSERT_STREQ(lhs().reads, expected);
    ASSERT_STREQ(rhs().reads, expected);
    Observer.Reset();
    HigherOrder::SplatSet(dst_ptr() + Offset, 5, Size);
    ASSERT_STREQ(src().reads, untouched);
    ASSERT_STREQ(src().writes, untouched);
    ASSERT_STREQ(dst().reads, untouched);
    ASSERT_STREQ(dst().writes, expected);
  }

  void checkMaxAccess(const BufferAccess &expected, int max) {
    for (size_t i = 0; i < kMaxBuffer; ++i) {
      int value = (int)expected[i] - '0';
      ASSERT_GE(value, 0);
      ASSERT_LE(value, max);
    }
  }

private:
  const Buffer &lhs() const { return Observer.Buffer1; }
  const Buffer &rhs() const { return Observer.Buffer2; }
  const Buffer &src() const { return Observer.Buffer2; }
  const Buffer &dst() const { return Observer.Buffer1; }
  Buffer &dst() { return Observer.Buffer1; }

  char *dst_ptr() { return dst().data.begin(); }
  const char *src_ptr() { return src().data.begin(); }
  const char *lhs_ptr() { return lhs().data.begin(); }
  const char *rhs_ptr() { return rhs().data.begin(); }
};

template <typename ParamType>
struct LlvmLibcTestAccessTail : public LlvmLibcTestAccessBase {

  void TearDown() override {
    static constexpr size_t Size = 10;

    BufferAccess expected;
    expected.Touch(Size - ParamType::kSize, ParamType::kSize);

    checkMaxAccess(expected, 1);
    checkOperations<Tail<ParamType>, Size>(expected);
  }
};
TYPED_TEST_F(LlvmLibcTestAccessTail, Operations, Types) {}

template <typename ParamType>
struct LlvmLibcTestAccessHeadTail : public LlvmLibcTestAccessBase {
  void TearDown() override {
    static constexpr size_t Size = 10;

    BufferAccess expected;
    expected.Touch(0, ParamType::kSize);
    expected.Touch(Size - ParamType::kSize, ParamType::kSize);

    checkMaxAccess(expected, 2);
    checkOperations<HeadTail<ParamType>, Size>(expected);
  }
};
TYPED_TEST_F(LlvmLibcTestAccessHeadTail, Operations, Types) {}

template <typename ParamType>
struct LlvmLibcTestAccessLoop : public LlvmLibcTestAccessBase {
  void TearDown() override {
    static constexpr size_t Size = 20;

    BufferAccess expected;
    for (size_t i = 0; i < Size - ParamType::kSize; i += ParamType::kSize)
      expected.Touch(i, ParamType::kSize);
    expected.Touch(Size - ParamType::kSize, ParamType::kSize);

    checkMaxAccess(expected, 2);
    checkOperations<Loop<ParamType>, Size>(expected);
  }
};
TYPED_TEST_F(LlvmLibcTestAccessLoop, Operations, Types) {}

template <typename ParamType>
struct LlvmLibcTestAccessAlignedAccess : public LlvmLibcTestAccessBase {
  void TearDown() override {
    static constexpr size_t Size = 10;
    static constexpr size_t Offset = 2;
    using AlignmentT = TestingElement<4>;

    BufferAccess expected;
    expected.Touch(Offset, AlignmentT::kSize);
    expected.Touch(AlignmentT::kSize, ParamType::kSize);
    expected.Touch(Offset + Size - ParamType::kSize, ParamType::kSize);

    checkMaxAccess(expected, 3);
    checkOperations<Align<AlignmentT, Arg::_1>::Then<HeadTail<ParamType>>, Size,
                    Offset>(expected);
    checkOperations<Align<AlignmentT, Arg::_2>::Then<HeadTail<ParamType>>, Size,
                    Offset>(expected);
  }
};
TYPED_TEST_F(LlvmLibcTestAccessAlignedAccess, Operations, Types) {}

} // namespace __llvm_libc
