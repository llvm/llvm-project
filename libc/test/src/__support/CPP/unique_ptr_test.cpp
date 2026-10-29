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

#include "hdr/signal_macros.h"
#include "src/__support/CPP/cstddef/nullptr_t.h"
#include "src/__support/CPP/unique_ptr.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/alloc-checker.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::AllocChecker;
using LIBC_NAMESPACE::cpp::nullptr_t;
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
    AllocChecker ac;
    unique_ptr<DestructTracker> ptr(new (ac) DestructTracker(&destruct_count));
    ASSERT_TRUE(static_cast<bool>(ac));
    ASSERT_TRUE(static_cast<bool>(ptr));
    ASSERT_EQ(destruct_count, 0);
  }
  ASSERT_EQ(destruct_count, 1);
}

// Test nullptr construction and behavior.
TEST(LlvmLibcUniquePtrTest, Nullptr) {
  unique_ptr<int> ptr(nullptr);
  ASSERT_FALSE(static_cast<bool>(ptr));
  ASSERT_EQ(ptr.get(), nullptr);

  constexpr nullptr_t null{};
  unique_ptr<int> ptr2(null);
  ASSERT_FALSE(static_cast<bool>(ptr2));
  ASSERT_EQ(ptr2.get(), nullptr);

  unique_ptr<int[]> arr(nullptr);
  ASSERT_FALSE(static_cast<bool>(arr));
  ASSERT_EQ(arr.get(), nullptr);

  unique_ptr<int[]> arr2(null);
  ASSERT_FALSE(static_cast<bool>(arr2));
  ASSERT_EQ(arr2.get(), nullptr);
}

// Test move construction transferring ownership.
TEST(LlvmLibcUniquePtrTest, Move) {
  int destruct_count = 0;
  {
    AllocChecker ac;
    unique_ptr<DestructTracker> ptr1(new (ac) DestructTracker(&destruct_count));
    ASSERT_TRUE(static_cast<bool>(ac));
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
    AllocChecker ac;
    unique_ptr<DestructTracker> ptr1(new (ac)
                                         DestructTracker(&destruct_count1));
    ASSERT_TRUE(static_cast<bool>(ac));
    unique_ptr<DestructTracker> ptr2(new (ac)
                                         DestructTracker(&destruct_count2));
    ASSERT_TRUE(static_cast<bool>(ac));
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
    AllocChecker ac;
    unique_ptr<DestructTracker> ptr(new (ac) DestructTracker(&destruct_count));
    ASSERT_TRUE(static_cast<bool>(ac));
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
    AllocChecker ac;
    unique_ptr<DestructTracker> ptr(new (ac) DestructTracker(&destruct_count1));
    ASSERT_TRUE(static_cast<bool>(ac));
    ptr.reset(new (ac) DestructTracker(&destruct_count2));
    ASSERT_TRUE(static_cast<bool>(ac));
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
  AllocChecker ac;
  unique_ptr<Foo> ptr(new (ac) Foo{42});
  ASSERT_TRUE(static_cast<bool>(ac));
  ASSERT_EQ((*ptr).val, 42);
  ASSERT_EQ(ptr->val, 42);
}

// Test array specialization behavior and destruction.
TEST(LlvmLibcUniquePtrTest, Array) {
  int destruct_count = 0;
  {
    AllocChecker ac;
    unique_ptr<DestructTracker[]> ptr(new (ac) DestructTracker[3]);
    ASSERT_TRUE(static_cast<bool>(ac));
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
    AllocChecker ac;
    unique_ptr<int, CustomDeleter> ptr(new (ac) int(42),
                                       CustomDeleter{&deleter_count});
    ASSERT_TRUE(static_cast<bool>(ac));
    ASSERT_EQ(deleter_count, 0);
  }
  ASSERT_EQ(deleter_count, 1);
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
    AllocChecker ac;
    CustomArrayDeleter d{&deleter_count};
    unique_ptr<int[], CustomArrayDeleter> ptr(new (ac) int[3], d);
    ASSERT_TRUE(static_cast<bool>(ac));
    ASSERT_EQ(deleter_count, 0);
    ASSERT_EQ(ptr.get_deleter().count, d.count);
  }
  ASSERT_EQ(deleter_count, 1);
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
  AllocChecker ac;
  ptr.reset(new (ac) ReentrancyTracker(&ptr, &checked));
  ASSERT_TRUE(static_cast<bool>(ac));
  ptr.reset(); // trigger destructor
  ASSERT_TRUE(checked);
}

// Test conversion constructor for const types.
TEST(LlvmLibcUniquePtrTest, ConstConversion) {
  AllocChecker ac;
  unique_ptr<int> ptr1(new (ac) int(42));
  ASSERT_TRUE(static_cast<bool>(ac));
  unique_ptr<const int> ptr2(LIBC_NAMESPACE::cpp::move(ptr1));
  ASSERT_FALSE(static_cast<bool>(ptr1));
  ASSERT_TRUE(static_cast<bool>(ptr2));
  ASSERT_EQ(*ptr2, 42);
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
  AllocChecker ac;
  unique_ptr<Derived> derived_ptr(new (ac) Derived(100));
  ASSERT_TRUE(static_cast<bool>(ac));
  unique_ptr<Base> base_ptr(LIBC_NAMESPACE::cpp::move(derived_ptr));
  ASSERT_FALSE(static_cast<bool>(derived_ptr));
  ASSERT_TRUE(static_cast<bool>(base_ptr));
  ASSERT_EQ(base_ptr->val, 100);
}

#ifndef NDEBUG
// In full builds, LIBC_ASSERT exits with code 0xFF. Otherwise, it uses the
// system assert, which raises SIGABRT.
#if defined(LIBC_FULL_BUILD) && !defined(LIBC_COPT_USE_C_ASSERT)
#define EXPECT_ASSERT_FAILURE(FUNC) EXPECT_EXITS(FUNC, 0xFF)
#else
#define EXPECT_ASSERT_FAILURE(FUNC) EXPECT_DEATH(FUNC, WITH_SIGNAL(SIGABRT))
#endif

// Test that dereferencing a null unique_ptr asserts.
TEST(LlvmLibcUniquePtrTest, NullDereferenceDeath) {
  struct Foo {
    int val;
  };
  unique_ptr<Foo> ptr;
  EXPECT_ASSERT_FAILURE([&] { (void)*ptr; });
  EXPECT_ASSERT_FAILURE([&] { (void)ptr->val; });

  unique_ptr<int[]> arr;
  EXPECT_ASSERT_FAILURE([&] { (void)arr[0]; });
}
#endif // NDEBUG
