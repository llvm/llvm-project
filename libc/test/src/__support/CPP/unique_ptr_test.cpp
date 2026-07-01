//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for UniquePtr.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/unique_ptr.h"
#include "src/__support/CPP/utility.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::unique_ptr;

struct DestructTracker {
  int *destruct_count = nullptr;
  DestructTracker() = default;
  DestructTracker(int *count) : destruct_count(count) {}
  ~DestructTracker() {
    if (destruct_count)
      (*destruct_count)++;
  }
};

// Test basic construction, ownership, and destruction.
TEST(LlvmLibcUniquePtrTest, Basic) {
  int destruct_count = 0;
  {
    unique_ptr<DestructTracker> ptr(new DestructTracker(&destruct_count));
    ASSERT_TRUE(static_cast<bool>(ptr));
    ASSERT_EQ(destruct_count, 0);
  }
  ASSERT_EQ(destruct_count, 1);
}

// Test nullptr construction and behavior.
TEST(LlvmLibcUniquePtrTest, Nullptr) {
  unique_ptr<int> ptr(nullptr);
  ASSERT_FALSE(static_cast<bool>(ptr));
  ASSERT_EQ(ptr.get(), static_cast<int *>(nullptr));
}

// Test move construction transferring ownership.
TEST(LlvmLibcUniquePtrTest, Move) {
  int destruct_count = 0;
  {
    unique_ptr<DestructTracker> ptr1(new DestructTracker(&destruct_count));
    unique_ptr<DestructTracker> ptr2(LIBC_NAMESPACE::cpp::move(ptr1));
    ASSERT_FALSE(static_cast<bool>(ptr1));
    ASSERT_TRUE(static_cast<bool>(ptr2));
    ASSERT_EQ(destruct_count, 0);
  }
  ASSERT_EQ(destruct_count, 1);
}

// Test move assignment transferring ownership and releasing old resource.
TEST(LlvmLibcUniquePtrTest, MoveAssignment) {
  int destruct_count1 = 0;
  int destruct_count2 = 0;
  {
    unique_ptr<DestructTracker> ptr1(new DestructTracker(&destruct_count1));
    unique_ptr<DestructTracker> ptr2(new DestructTracker(&destruct_count2));
    ptr2 = LIBC_NAMESPACE::cpp::move(ptr1);
    ASSERT_FALSE(static_cast<bool>(ptr1));
    ASSERT_TRUE(static_cast<bool>(ptr2));
    ASSERT_EQ(destruct_count1, 0);
    ASSERT_EQ(destruct_count2, 1); // ptr2's original object should be destroyed
  }
  ASSERT_EQ(destruct_count1, 1);
}

// Test release of ownership without destroying the object.
TEST(LlvmLibcUniquePtrTest, Release) {
  int destruct_count = 0;
  DestructTracker *raw_ptr = nullptr;
  {
    unique_ptr<DestructTracker> ptr(new DestructTracker(&destruct_count));
    raw_ptr = ptr.release();
    ASSERT_FALSE(static_cast<bool>(ptr));
    ASSERT_EQ(destruct_count, 0);
  }
  ASSERT_EQ(destruct_count, 0);
  delete raw_ptr;
  ASSERT_EQ(destruct_count, 1);
}

// Test reset replacing the owned object and destroying the old one.
TEST(LlvmLibcUniquePtrTest, Reset) {
  int destruct_count1 = 0;
  int destruct_count2 = 0;
  {
    unique_ptr<DestructTracker> ptr(new DestructTracker(&destruct_count1));
    ptr.reset(new DestructTracker(&destruct_count2));
    ASSERT_EQ(destruct_count1, 1);
    ASSERT_EQ(destruct_count2, 0);
  }
  ASSERT_EQ(destruct_count2, 1);
}

// Test dereference operators (operator* and operator->).
TEST(LlvmLibcUniquePtrTest, Dereference) {
  struct Foo {
    int val;
  };
  unique_ptr<Foo> ptr(new Foo{42});
  ASSERT_EQ((*ptr).val, 42);
  ASSERT_EQ(ptr->val, 42);
}

// Test array specialization behavior and destruction.
TEST(LlvmLibcUniquePtrTest, Array) {
  int destruct_count = 0;
  {
    unique_ptr<DestructTracker[]> ptr(new DestructTracker[3]);
    ptr[0].destruct_count = &destruct_count;
    ptr[1].destruct_count = &destruct_count;
    ptr[2].destruct_count = &destruct_count;
    ASSERT_TRUE(static_cast<bool>(ptr));
    ASSERT_EQ(destruct_count, 0);
  }
  ASSERT_EQ(destruct_count, 3);
}

struct CustomDeleter {
  int *count;
  void operator()(int *p) const {
    (*count)++;
    delete p;
  }
};

// Test support for custom deleters.
TEST(LlvmLibcUniquePtrTest, CustomDeleter) {
  int deleter_count = 0;
  {
    unique_ptr<int, CustomDeleter> ptr(new int(42),
                                       CustomDeleter{&deleter_count});
    ASSERT_EQ(deleter_count, 0);
  }
  ASSERT_EQ(deleter_count, 1);
}
