//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for cpp::unique_ptr.
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
    ASSERT_NE(ptr.get(), nullptr);
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 1);
}

// Test nullptr construction and behavior.
TEST(LlvmLibcUniquePtrTest, Nullptr) {
  unique_ptr<int> ptr(nullptr);
  EXPECT_EQ(ptr.get(), nullptr);
  EXPECT_FALSE(static_cast<bool>(ptr));
  EXPECT_TRUE(ptr == nullptr);
  EXPECT_TRUE(nullptr == ptr);
}

// Test move construction transferring ownership.
TEST(LlvmLibcUniquePtrTest, Move) {
  int destruct_count = 0;
  {
    unique_ptr<DestructTracker> ptr1(new DestructTracker(&destruct_count));
    unique_ptr<DestructTracker> ptr2(LIBC_NAMESPACE::cpp::move(ptr1));
    EXPECT_EQ(ptr1.get(), nullptr);
    ASSERT_NE(ptr2.get(), nullptr);
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 1);
}

// Test move assignment transferring ownership and releasing old resource.
TEST(LlvmLibcUniquePtrTest, MoveAssignment) {
  int destruct_count1 = 0;
  int destruct_count2 = 0;
  {
    unique_ptr<DestructTracker> ptr1(new DestructTracker(&destruct_count1));
    unique_ptr<DestructTracker> ptr2(new DestructTracker(&destruct_count2));
    ptr2 = LIBC_NAMESPACE::cpp::move(ptr1);
    EXPECT_EQ(ptr1.get(), nullptr);
    ASSERT_NE(ptr2.get(), nullptr);
    EXPECT_EQ(destruct_count1, 0);
    EXPECT_EQ(destruct_count2, 1);
  }
  EXPECT_EQ(destruct_count1, 1);
}

// Test self move-assignment.
TEST(LlvmLibcUniquePtrTest, SelfMoveAssignment) {
  int destruct_count = 0;
  {
    unique_ptr<DestructTracker> ptr(new DestructTracker(&destruct_count));
    ptr = LIBC_NAMESPACE::cpp::move(ptr);
    ASSERT_NE(ptr.get(), nullptr);
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 1);
}

// Test nullptr assignment.
TEST(LlvmLibcUniquePtrTest, NullptrAssignment) {
  int destruct_count = 0;
  unique_ptr<DestructTracker> ptr(new DestructTracker(&destruct_count));
  ASSERT_NE(ptr.get(), nullptr);
  ptr = nullptr;
  EXPECT_EQ(ptr.get(), nullptr);
  EXPECT_EQ(destruct_count, 1);
}

// Test release of ownership without destroying the object.
TEST(LlvmLibcUniquePtrTest, Release) {
  int destruct_count = 0;
  DestructTracker *raw_ptr = nullptr;
  {
    unique_ptr<DestructTracker> ptr(new DestructTracker(&destruct_count));
    raw_ptr = ptr.release();
    EXPECT_EQ(ptr.get(), nullptr);
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 0);
  ASSERT_NE(raw_ptr, nullptr);
  delete raw_ptr;
  EXPECT_EQ(destruct_count, 1);
}

// Test reset replacing the owned object and destroying the old one.
TEST(LlvmLibcUniquePtrTest, Reset) {
  int destruct_count1 = 0;
  int destruct_count2 = 0;
  {
    unique_ptr<DestructTracker> ptr(new DestructTracker(&destruct_count1));
    ptr.reset(new DestructTracker(&destruct_count2));
    EXPECT_EQ(destruct_count1, 1);
    EXPECT_EQ(destruct_count2, 0);
    ptr.reset();
    EXPECT_EQ(destruct_count2, 1);
    EXPECT_EQ(ptr.get(), nullptr);
  }
}

// Test dereference operators (operator* and operator->).
TEST(LlvmLibcUniquePtrTest, Dereference) {
  struct Foo {
    int val;
  };
  unique_ptr<Foo> ptr(new Foo{42});
  ASSERT_NE(ptr.get(), nullptr);
  EXPECT_EQ((*ptr).val, 42);
  EXPECT_EQ(ptr->val, 42);
}

// Test swap member and non-member function.
TEST(LlvmLibcUniquePtrTest, Swap) {
  unique_ptr<int> ptr1(new int(1));
  unique_ptr<int> ptr2(new int(2));
  int *p1 = ptr1.get();
  int *p2 = ptr2.get();

  ptr1.swap(ptr2);
  EXPECT_EQ(ptr1.get(), p2);
  EXPECT_EQ(ptr2.get(), p1);
  EXPECT_EQ(*ptr1, 2);
  EXPECT_EQ(*ptr2, 1);

  LIBC_NAMESPACE::cpp::swap(ptr1, ptr2);
  EXPECT_EQ(ptr1.get(), p1);
  EXPECT_EQ(ptr2.get(), p2);
  EXPECT_EQ(*ptr1, 1);
  EXPECT_EQ(*ptr2, 2);
}

// Test equality and inequality comparisons.
TEST(LlvmLibcUniquePtrTest, Comparisons) {
  unique_ptr<int> ptr1(new int(10));
  unique_ptr<int> ptr2(new int(20));
  unique_ptr<int> null_ptr;

  EXPECT_TRUE(ptr1 == ptr1);
  EXPECT_FALSE(ptr1 == ptr2);
  EXPECT_TRUE(ptr1 != ptr2);

  EXPECT_TRUE(null_ptr == nullptr);
  EXPECT_TRUE(nullptr == null_ptr);
  EXPECT_FALSE(ptr1 == nullptr);
  EXPECT_FALSE(nullptr == ptr1);

  EXPECT_FALSE(null_ptr != nullptr);
  EXPECT_FALSE(nullptr != null_ptr);
  EXPECT_TRUE(ptr1 != nullptr);
  EXPECT_TRUE(nullptr != ptr1);
}

// Test array specialization behavior and destruction.
TEST(LlvmLibcUniquePtrTest, Array) {
  int destruct_count = 0;
  {
    unique_ptr<DestructTracker[]> ptr(new DestructTracker[3]);
    ASSERT_NE(ptr.get(), nullptr);
    ptr[0].destruct_count = &destruct_count;
    ptr[1].destruct_count = &destruct_count;
    ptr[2].destruct_count = &destruct_count;
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 3);
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
    EXPECT_EQ(deleter_count, 0);
  }
  EXPECT_EQ(deleter_count, 1);
}

struct CustomArrayDeleter {
  int *count;
  void operator()(int *p) const {
    if (count)
      (*count)++;
    delete[] p;
  }
};

// Test support for custom deleters on array unique_ptr.
TEST(LlvmLibcUniquePtrTest, CustomArrayDeleter) {
  int deleter_count = 0;
  {
    CustomArrayDeleter d{&deleter_count};
    unique_ptr<int[], CustomArrayDeleter> ptr(new int[3], d);
    EXPECT_EQ(deleter_count, 0);
    EXPECT_EQ(ptr.get_deleter().count, d.count);
  }
  EXPECT_EQ(deleter_count, 1);
}

struct ReentrancyTracker {
  unique_ptr<ReentrancyTracker> *owner;
  bool *checked;
  ReentrancyTracker(unique_ptr<ReentrancyTracker> *o, bool *c)
      : owner(o), checked(c) {}
  ~ReentrancyTracker() {
    if (owner)
      *checked = (owner->get() == nullptr);
  }
};

// Test that reset() clears the pointer before invoking the deleter.
TEST(LlvmLibcUniquePtrTest, ResetSequencing) {
  bool checked = false;
  unique_ptr<ReentrancyTracker> ptr;
  ptr.reset(new ReentrancyTracker(&ptr, &checked));
  ptr.reset(); // trigger destructor
  EXPECT_TRUE(checked);
}

// Test conversion constructor for const types.
TEST(LlvmLibcUniquePtrTest, ConstConversion) {
  unique_ptr<int> ptr1(new int(42));
  unique_ptr<const int> ptr2(LIBC_NAMESPACE::cpp::move(ptr1));
  EXPECT_EQ(ptr1.get(), nullptr);
  ASSERT_NE(ptr2.get(), nullptr);
  EXPECT_EQ(*ptr2, 42);
}

struct Base {
  int val;
  virtual ~Base() = default;
  Base(int v) : val(v) {}
};
struct Derived : public Base {
  Derived(int v) : Base(v) {}
};

// Test conversion constructor for base/derived types.
TEST(LlvmLibcUniquePtrTest, BaseDerivedConversion) {
  unique_ptr<Derived> derived_ptr(new Derived(100));
  unique_ptr<Base> base_ptr(LIBC_NAMESPACE::cpp::move(derived_ptr));
  EXPECT_EQ(derived_ptr.get(), nullptr);
  ASSERT_NE(base_ptr.get(), nullptr);
  EXPECT_EQ(base_ptr->val, 100);
}

// Test conversion assignment for base/derived types.
TEST(LlvmLibcUniquePtrTest, BaseDerivedAssignment) {
  unique_ptr<Derived> derived_ptr(new Derived(200));
  unique_ptr<Base> base_ptr;
  base_ptr = LIBC_NAMESPACE::cpp::move(derived_ptr);
  EXPECT_EQ(derived_ptr.get(), nullptr);
  ASSERT_NE(base_ptr.get(), nullptr);
  EXPECT_EQ(base_ptr->val, 200);
}

// Test move construction for array unique_ptr.
TEST(LlvmLibcUniquePtrTest, ArrayMove) {
  int destruct_count = 0;
  {
    unique_ptr<DestructTracker[]> ptr1(new DestructTracker[2]);
    ASSERT_NE(ptr1.get(), nullptr);
    ptr1[0].destruct_count = &destruct_count;
    ptr1[1].destruct_count = &destruct_count;
    unique_ptr<DestructTracker[]> ptr2(LIBC_NAMESPACE::cpp::move(ptr1));
    EXPECT_EQ(ptr1.get(), nullptr);
    ASSERT_NE(ptr2.get(), nullptr);
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 2);
}

// Test move assignment for array unique_ptr.
TEST(LlvmLibcUniquePtrTest, ArrayMoveAssignment) {
  int destruct_count1 = 0;
  int destruct_count2 = 0;
  {
    unique_ptr<DestructTracker[]> ptr1(new DestructTracker[2]);
    ASSERT_NE(ptr1.get(), nullptr);
    ptr1[0].destruct_count = &destruct_count1;
    ptr1[1].destruct_count = &destruct_count1;
    unique_ptr<DestructTracker[]> ptr2(new DestructTracker[3]);
    ASSERT_NE(ptr2.get(), nullptr);
    ptr2[0].destruct_count = &destruct_count2;
    ptr2[1].destruct_count = &destruct_count2;
    ptr2[2].destruct_count = &destruct_count2;
    ptr2 = LIBC_NAMESPACE::cpp::move(ptr1);
    EXPECT_EQ(ptr1.get(), nullptr);
    ASSERT_NE(ptr2.get(), nullptr);
    EXPECT_EQ(destruct_count1, 0);
    EXPECT_EQ(destruct_count2, 3);
  }
  EXPECT_EQ(destruct_count1, 2);
}

// Test array reset and release.
TEST(LlvmLibcUniquePtrTest, ArrayResetAndRelease) {
  int destruct_count1 = 0;
  int destruct_count2 = 0;
  {
    unique_ptr<DestructTracker[]> ptr(new DestructTracker[2]);
    ASSERT_NE(ptr.get(), nullptr);
    ptr[0].destruct_count = &destruct_count1;
    ptr[1].destruct_count = &destruct_count1;

    DestructTracker *new_arr = new DestructTracker[2];
    ASSERT_NE(new_arr, nullptr);
    new_arr[0].destruct_count = &destruct_count2;
    new_arr[1].destruct_count = &destruct_count2;
    ptr.reset(new_arr);
    EXPECT_EQ(destruct_count1, 2);
    EXPECT_EQ(destruct_count2, 0);

    DestructTracker *released = ptr.release();
    EXPECT_EQ(ptr.get(), nullptr);
    EXPECT_EQ(destruct_count2, 0);
    ASSERT_NE(released, nullptr);
    delete[] released;
    EXPECT_EQ(destruct_count2, 2);
  }
}

// Test array swap.
TEST(LlvmLibcUniquePtrTest, ArraySwap) {
  unique_ptr<int[]> ptr1(new int[2]{1, 2});
  unique_ptr<int[]> ptr2(new int[2]{3, 4});
  int *p1 = ptr1.get();
  int *p2 = ptr2.get();

  ptr1.swap(ptr2);
  EXPECT_EQ(ptr1.get(), p2);
  EXPECT_EQ(ptr2.get(), p1);
  EXPECT_EQ(ptr1[0], 3);
  EXPECT_EQ(ptr2[0], 1);

  LIBC_NAMESPACE::cpp::swap(ptr1, ptr2);
  EXPECT_EQ(ptr1.get(), p1);
  EXPECT_EQ(ptr2.get(), p2);
  EXPECT_EQ(ptr1[0], 1);
  EXPECT_EQ(ptr2[0], 3);
}

// Test empty deleter size optimization (LIBC_NO_UNIQUE_ADDRESS).
TEST(LlvmLibcUniquePtrTest, EmptyDeleterOptimization) {
  static_assert(
      sizeof(unique_ptr<int>) == sizeof(int *),
      "unique_ptr with empty deleter must be same size as raw pointer");
  static_assert(
      sizeof(unique_ptr<int[]>) == sizeof(int *),
      "unique_ptr[] with empty deleter must be same size as raw pointer");
}
