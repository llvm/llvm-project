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
#include "sanitizer_common/sanitizer_vector.h"
#include "gtest/gtest.h"

namespace __sanitizer {

TEST(Vector, Basic) {
  Vector<int> v;
  EXPECT_EQ(v.Size(), (uptr)0);
  v.PushBack(42);
  EXPECT_EQ(v.Size(), (uptr)1);
  EXPECT_EQ(v[0], 42);
  v.PushBack(43);
  EXPECT_EQ(v.Size(), (uptr)2);
  EXPECT_EQ(v[0], 42);
  EXPECT_EQ(v[1], 43);
}

TEST(Vector, Stride) {
  Vector<int> v;
  for (int i = 0; i < 1000; i++) {
    v.PushBack(i);
    EXPECT_EQ(v.Size(), (uptr)(i + 1));
    EXPECT_EQ(v[i], i);
  }
  for (int i = 0; i < 1000; i++) {
    EXPECT_EQ(v[i], i);
  }
}

}  // namespace __sanitizer
