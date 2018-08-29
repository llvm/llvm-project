//===-- sanitizer_vector_test.cc ------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of *Sanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_ring_buffer.h"
#include "gtest/gtest.h"

namespace __sanitizer {

struct LargeStruct {
  long v;
  long extra[3];

  explicit LargeStruct(long v) : v(v) {}
  operator long() { return v; }
};

struct Struct10Bytes {
  short t[3];
};

TEST(RingBuffer, Construct) {
  RingBuffer<long> *RBlong = RingBuffer<long>::New(20);
  EXPECT_EQ(RBlong->size(), 20U);
  RBlong->Delete();
}

template <class T> void TestRB() {
  RingBuffer<T> *RB;
  const size_t Sizes[] = {1, 2, 3, 5, 8, 16, 20, 40, 10000};
  for (size_t Size : Sizes) {
    RB = RingBuffer<T>::New(Size);
    EXPECT_EQ(RB->size(), Size);
    RB->Delete();
  }

  RB = RingBuffer<T>::New(4);
  EXPECT_EQ(RB->size(), 4U);
#define EXPECT_RING_BUFFER(a0, a1, a2, a3) \
  EXPECT_EQ((long)(*RB)[0], (long)a0);                 \
  EXPECT_EQ((long)(*RB)[1], (long)a1);                 \
  EXPECT_EQ((long)(*RB)[2], (long)a2);                 \
  EXPECT_EQ((long)(*RB)[3], (long)a3);

  RB->push(T(1)); EXPECT_RING_BUFFER(1, 0, 0, 0);
  RB->push(T(2)); EXPECT_RING_BUFFER(2, 1, 0, 0);
  RB->push(T(3)); EXPECT_RING_BUFFER(3, 2, 1, 0);
  RB->push(T(4)); EXPECT_RING_BUFFER(4, 3, 2, 1);
  RB->push(T(5)); EXPECT_RING_BUFFER(5, 4, 3, 2);
  RB->push(T(6)); EXPECT_RING_BUFFER(6, 5, 4, 3);
  RB->push(T(7)); EXPECT_RING_BUFFER(7, 6, 5, 4);
  RB->push(T(8)); EXPECT_RING_BUFFER(8, 7, 6, 5);
  RB->push(T(9)); EXPECT_RING_BUFFER(9, 8, 7, 6);
  RB->push(T(10)); EXPECT_RING_BUFFER(10, 9, 8, 7);
  RB->push(T(11)); EXPECT_RING_BUFFER(11, 10, 9, 8);
  RB->push(T(12)); EXPECT_RING_BUFFER(12, 11, 10, 9);

#undef EXPECT_RING_BUFFER
}

TEST(RingBuffer, Int) {
  EXPECT_DEATH(RingBuffer<short>::New(10), "");
  EXPECT_DEATH(RingBuffer<Struct10Bytes>::New(10), "");
  if (sizeof(int) < sizeof(void *))
    EXPECT_DEATH(RingBuffer<int>::New(10), "");
}

TEST(RingBuffer, Long) {
  TestRB<long>();
}

TEST(RingBuffer, LargeStruct) {
  TestRB<LargeStruct>();
}

}  // namespace __sanitizer
