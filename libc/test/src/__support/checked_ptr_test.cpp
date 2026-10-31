//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for CheckedPtr and make_unique.
///
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/unique_ptr.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/checked_ptr.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::AllocChecker;
using LIBC_NAMESPACE::CheckedPtr;
using LIBC_NAMESPACE::make_unique;
using LIBC_NAMESPACE::cpp::is_constructible_v;
using LIBC_NAMESPACE::cpp::move;
using LIBC_NAMESPACE::cpp::unique_ptr;

// Verify compile-time properties:
// - Non-copyable and non-default-constructible.
// - Explicit conversion to bool is only allowed on non-const lvalues/rvalues.
static_assert(!is_constructible_v<CheckedPtr<int>>);
static_assert(!is_constructible_v<CheckedPtr<int>, const CheckedPtr<int> &>);
static_assert(is_constructible_v<bool, CheckedPtr<int> &>);
static_assert(!is_constructible_v<bool, const CheckedPtr<int> &>);
static_assert(is_constructible_v<bool, CheckedPtr<int[]> &>);
static_assert(!is_constructible_v<bool, const CheckedPtr<int[]> &>);

struct DestructTracker {
  int *destruct_count = nullptr;
  DestructTracker() = default;
  DestructTracker(int *count) : destruct_count(count) {}
  ~DestructTracker() {
    if (destruct_count)
      (*destruct_count)++;
  }
};

struct Foo {
  int val;
  Foo(int v) : val(v) {}
};

struct Base {
  int val;
  virtual ~Base() = default;
  Base(int v) : val(v) {}
};

struct Derived : public Base {
  Derived(int v) : Base(v) {}
};

struct CustomDeleter {
  int *count;
  void operator()(int *p) const {
    (*count)++;
    delete p;
  }
};

struct CustomArrayDeleter {
  int *count;
  void operator()(int *p) const {
    if (count)
      (*count)++;
    delete[] p;
  }
};

TEST(LlvmLibcCheckedPtrTest, BasicMakeUniqueAndAccess) {
  int destruct_count = 0;
  {
    auto ptr = make_unique<DestructTracker>(&destruct_count);
    ASSERT_TRUE(static_cast<bool>(ptr));

    // Once checked, access via const reference is valid and safe.
    const auto &cref = ptr;
    EXPECT_EQ(*cref->destruct_count, 0);
    EXPECT_EQ(*(*cref).destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 1);
}

TEST(LlvmLibcCheckedPtrTest, NullptrCheck) {
  CheckedPtr<int> ptr(nullptr);
  EXPECT_FALSE(static_cast<bool>(ptr));
}

TEST(LlvmLibcCheckedPtrTest, MoveRequiresRecheck) {
  int destruct_count = 0;
  {
    auto ptr1 = make_unique<DestructTracker>(&destruct_count);
    EXPECT_TRUE(static_cast<bool>(ptr1));

    // Even though ptr1 was checked, moving into ptr2 requires checking ptr2.
    CheckedPtr<DestructTracker> ptr2(move(ptr1));
    ASSERT_TRUE(static_cast<bool>(ptr2));
    EXPECT_EQ(*ptr2->destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 1);
}

TEST(LlvmLibcCheckedPtrTest, MoveUncheckedThenCheck) {
  int destruct_count = 0;
  {
    auto ptr1 = make_unique<DestructTracker>(&destruct_count);
    // Move before checking: ptr1 is disarmed; ptr2 must be checked.
    CheckedPtr<DestructTracker> ptr2(move(ptr1));
    ASSERT_TRUE(static_cast<bool>(ptr2));
    EXPECT_EQ(*ptr2->destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 1);
}

TEST(LlvmLibcCheckedPtrTest, MoveAssignment) {
  int destruct_count1 = 0;
  int destruct_count2 = 0;
  {
    auto ptr1 = make_unique<DestructTracker>(&destruct_count1);
    EXPECT_TRUE(static_cast<bool>(ptr1));
    auto ptr2 = make_unique<DestructTracker>(&destruct_count2);
    EXPECT_TRUE(static_cast<bool>(ptr2));

    ptr2 = move(ptr1);
    EXPECT_EQ(destruct_count1, 0);
    EXPECT_EQ(destruct_count2, 1);
    EXPECT_TRUE(static_cast<bool>(ptr2));
  }
  EXPECT_EQ(destruct_count1, 1);
}

TEST(LlvmLibcCheckedPtrTest, Conversions) {
  auto derived = make_unique<Derived>(100);
  CheckedPtr<Base> base(move(derived));
  ASSERT_TRUE(static_cast<bool>(base));
  EXPECT_EQ(base->val, 100);

  auto mut_int = make_unique<int>(42);
  CheckedPtr<const int> const_int(move(mut_int));
  ASSERT_TRUE(static_cast<bool>(const_int));
  EXPECT_EQ(*const_int, 42);
}

TEST(LlvmLibcCheckedPtrTest, Array) {
  int destruct_count = 0;
  {
    auto ptr = make_unique<DestructTracker[]>(3);
    ASSERT_TRUE(static_cast<bool>(ptr));
    ptr[0].destruct_count = &destruct_count;
    ptr[1].destruct_count = &destruct_count;
    ptr[2].destruct_count = &destruct_count;
    EXPECT_EQ(destruct_count, 0);
  }
  EXPECT_EQ(destruct_count, 3);
}

TEST(LlvmLibcCheckedPtrTest, CustomDeleters) {
  int obj_deletes = 0;
  {
    AllocChecker ac;
    CheckedPtr<int, CustomDeleter> ptr(new (ac) int(7),
                                       CustomDeleter{&obj_deletes});
    ASSERT_TRUE(static_cast<bool>(ptr));
    EXPECT_EQ(*ptr, 7);
    EXPECT_EQ(obj_deletes, 0);
  }
  EXPECT_EQ(obj_deletes, 1);

  int arr_deletes = 0;
  {
    AllocChecker ac;
    CheckedPtr<int[], CustomArrayDeleter> arr(new (ac) int[2],
                                              CustomArrayDeleter{&arr_deletes});
    ASSERT_TRUE(static_cast<bool>(arr));
    arr[0] = 10;
    arr[1] = 20;
    EXPECT_EQ(arr[0] + arr[1], 30);
    EXPECT_EQ(arr_deletes, 0);
  }
  EXPECT_EQ(arr_deletes, 1);
}

TEST(LlvmLibcCheckedPtrTest, SelfSwap) {
  auto ptr = make_unique<int>(1);
  EXPECT_TRUE(static_cast<bool>(ptr));
  // Self-move-assignment (which happens as a part of self-swap) must be a
  // no-op.
  LIBC_NAMESPACE::cpp::swap(ptr, ptr);
  ASSERT_TRUE(static_cast<bool>(ptr));
  EXPECT_EQ(*ptr, 1);
}

struct VoidDeleter {
  void operator()(void *) const {}
};

TEST(LlvmLibcCheckedPtrTest, Void) {
  CheckedPtr<void, VoidDeleter> ptr(nullptr);
  EXPECT_FALSE(static_cast<bool>(ptr));
}

TEST(LlvmLibcCheckedPtrTest, TakeUnique) {
  int destruct_count = 0;
  unique_ptr<DestructTracker> owner;
  {
    auto ptr = make_unique<DestructTracker>(&destruct_count);
    EXPECT_TRUE(static_cast<bool>(ptr));
    owner = ptr.take_unique();
    // `ptr` is now null and still checked, so destroying it must neither delete
    // the object nor assert.
  }
  EXPECT_EQ(destruct_count, 0);
  EXPECT_TRUE(static_cast<bool>(owner));
  owner.reset();
  EXPECT_EQ(destruct_count, 1);
}

#if !defined(NDEBUG) && LIBC_TEST_SUBPROCESS_TESTS

#if defined(LIBC_FULL_BUILD) && !defined(LIBC_COPT_USE_C_ASSERT)
#define EXPECT_ASSERT_FAILS(stmt) EXPECT_EXITS(stmt, 255)
#else
#define EXPECT_ASSERT_FAILS(stmt) EXPECT_DEATH(stmt, WITH_SIGNAL(-1))
#endif

TEST(LlvmLibcCheckedPtrDeathTest, UncheckedDestructionDies) {
  EXPECT_ASSERT_FAILS([] {
    auto ptr = make_unique<int>(42);
    (void)ptr;
  });
}

TEST(LlvmLibcCheckedPtrDeathTest, MoveFromCheckedRequiresRecheckOrDies) {
  EXPECT_ASSERT_FAILS([] {
    auto ptr1 = make_unique<int>(42);
    (void)static_cast<bool>(ptr1);
    CheckedPtr<int> ptr2(move(ptr1));
    (void)*ptr2;
  });
}

// In the tests below, the pointer is checked after the operation under test, so
// that its destruction cannot be the thing that triggers the assertion.

TEST(LlvmLibcCheckedPtrDeathTest, UncheckedDereferenceDies) {
  EXPECT_ASSERT_FAILS([] {
    auto ptr = make_unique<int>(42);
    (void)*ptr;
    (void)static_cast<bool>(ptr);
  });
}

TEST(LlvmLibcCheckedPtrDeathTest, UncheckedArrowDies) {
  EXPECT_ASSERT_FAILS([] {
    auto ptr = make_unique<Foo>(42);
    (void)ptr->val;
    (void)static_cast<bool>(ptr);
  });
}

TEST(LlvmLibcCheckedPtrDeathTest, UncheckedArraySubscriptDies) {
  EXPECT_ASSERT_FAILS([] {
    auto ptr = make_unique<int[]>(3);
    (void)ptr[0];
    (void)static_cast<bool>(ptr);
  });
}

TEST(LlvmLibcCheckedPtrDeathTest, OverwritingUncheckedViaMoveAssignmentDies) {
  EXPECT_ASSERT_FAILS([] {
    auto ptr1 = make_unique<int>(1);
    auto ptr2 = make_unique<int>(2);
    (void)static_cast<bool>(ptr2);
    ptr1 = move(ptr2);
    (void)static_cast<bool>(ptr1);
  });
}

TEST(LlvmLibcCheckedPtrDeathTest, UncheckedTakeUniqueDies) {
  EXPECT_ASSERT_FAILS([] {
    auto ptr = make_unique<int>(42);
    (void)ptr.take_unique();
    (void)static_cast<bool>(ptr);
  });
}

#endif // !defined(NDEBUG) && LIBC_TEST_SUBPROCESS_TESTS
