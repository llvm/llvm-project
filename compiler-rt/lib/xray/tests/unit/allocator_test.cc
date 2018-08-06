//===-- allocator_test.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a function call tracing system.
//
//===----------------------------------------------------------------------===//

#include "xray_allocator.h"
#include "gtest/gtest.h"

namespace __xray {
namespace {

struct TestData {
  s64 First;
  s64 Second;
};

TEST(AllocatorTest, Construction) { Allocator<sizeof(TestData)> A(2 << 11); }

TEST(AllocatorTest, Allocate) {
  Allocator<sizeof(TestData)> A(2 << 11);
  auto B = A.Allocate();
  ASSERT_NE(B.Data, nullptr);
}

TEST(AllocatorTest, OverAllocate) {
  Allocator<sizeof(TestData)> A(sizeof(TestData));
  auto B1 = A.Allocate();
  (void)B1;
  auto B2 = A.Allocate();
  ASSERT_EQ(B2.Data, nullptr);
}

} // namespace
} // namespace __xray
