//===-- Unittests for ArrayRef --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/ArrayRef.h"
#include "utils/UnitTest/Test.h"

namespace __llvm_libc {
namespace cpp {

// The following tests run on both 'ArrayRef' and 'MutableArrayRef'.
using Types = testing::TypeList<ArrayRef<int>, MutableArrayRef<int>>;

TYPED_TEST(LlvmLibcArrayRefTest, ConstructFromElement, Types) {
  using value_type = typename ParamType::value_type;
  using const_pointer = typename ParamType::const_pointer;
  value_type element = 5;
  ParamType arrayref(element);
  EXPECT_FALSE(arrayref.empty());
  EXPECT_EQ(arrayref.size(), size_t(1));
  EXPECT_EQ(arrayref[0], 5);
  EXPECT_EQ((const_pointer)arrayref.data(), (const_pointer)&element);
}

TYPED_TEST(LlvmLibcArrayRefTest, ConstructFromPointerAndSize, Types) {
  using value_type = typename ParamType::value_type;
  using const_pointer = typename ParamType::const_pointer;
  value_type values[] = {1, 2};
  ParamType arrayref(values, 2);
  EXPECT_FALSE(arrayref.empty());
  EXPECT_EQ(arrayref.size(), size_t(2));
  EXPECT_EQ(arrayref[0], 1);
  EXPECT_EQ(arrayref[1], 2);
  EXPECT_EQ((const_pointer)arrayref.data(), (const_pointer)values);
}

TYPED_TEST(LlvmLibcArrayRefTest, ConstructFromIterator, Types) {
  using value_type = typename ParamType::value_type;
  using const_pointer = typename ParamType::const_pointer;
  value_type values[] = {1, 2};
  ParamType arrayref(&values[0], &values[2]);
  EXPECT_FALSE(arrayref.empty());
  EXPECT_EQ(arrayref.size(), size_t(2));
  EXPECT_EQ(arrayref[0], 1);
  EXPECT_EQ(arrayref[1], 2);
  EXPECT_EQ((const_pointer)arrayref.data(), (const_pointer)&values[0]);
}

TYPED_TEST(LlvmLibcArrayRefTest, ConstructFromCArray, Types) {
  using value_type = typename ParamType::value_type;
  using const_pointer = typename ParamType::const_pointer;
  value_type values[] = {1, 2};
  ParamType arrayref(values);
  EXPECT_FALSE(arrayref.empty());
  EXPECT_EQ(arrayref.size(), size_t(2));
  EXPECT_EQ(arrayref[0], 1);
  EXPECT_EQ(arrayref[1], 2);
  EXPECT_EQ((const_pointer)arrayref.data(), (const_pointer)values);
}

TYPED_TEST(LlvmLibcArrayRefTest, ConstructFromLibcArray, Types) {
  using value_type = typename ParamType::value_type;
  using const_pointer = typename ParamType::const_pointer;
  Array<value_type, 2> values = {1, 2};
  ParamType arrayref(values);
  EXPECT_FALSE(arrayref.empty());
  EXPECT_EQ(arrayref.size(), size_t(2));
  EXPECT_EQ(arrayref[0], 1);
  EXPECT_EQ(arrayref[1], 2);
  EXPECT_EQ((const_pointer)arrayref.data(), (const_pointer)values.data());
}

TYPED_TEST(LlvmLibcArrayRefTest, Equals, Types) {
  using value_type = typename ParamType::value_type;
  value_type values[] = {1, 2, 3};
  ParamType initial(values);
  EXPECT_TRUE(initial.equals(initial));
  ParamType shallow_copy(values);
  EXPECT_TRUE(initial.equals(shallow_copy));
  value_type same_values[] = {1, 2, 3};
  EXPECT_TRUE(initial.equals(same_values));
  value_type different_values[] = {1, 2, 4};
  EXPECT_FALSE(initial.equals(different_values));
}

TYPED_TEST(LlvmLibcArrayRefTest, SliceUnary, Types) {
  using value_type = typename ParamType::value_type;
  value_type values[] = {1, 2, 3};
  ParamType arrayref(values);
  {
    value_type values[] = {1, 2, 3};
    EXPECT_TRUE(arrayref.slice(0).equals(values));
  }
  {
    value_type values[] = {2, 3};
    EXPECT_TRUE(arrayref.slice(1).equals(values));
  }
  {
    value_type values[] = {3};
    EXPECT_TRUE(arrayref.slice(2).equals(values));
  }
  { EXPECT_TRUE(arrayref.slice(3).empty()); }
}

TYPED_TEST(LlvmLibcArrayRefTest, SliceBinary, Types) {
  using value_type = typename ParamType::value_type;
  value_type values[] = {1, 2, 3};
  ParamType arrayref(values);
  {
    EXPECT_TRUE(arrayref.slice(0, 0).empty());
    EXPECT_TRUE(arrayref.slice(1, 0).empty());
    EXPECT_TRUE(arrayref.slice(2, 0).empty());
    EXPECT_TRUE(arrayref.slice(3, 0).empty());
  }
  {
    value_type values[] = {1};
    EXPECT_TRUE(arrayref.slice(0, 1).equals(values));
  }
  {
    value_type values[] = {2};
    EXPECT_TRUE(arrayref.slice(1, 1).equals(values));
  }
  {
    value_type values[] = {3};
    EXPECT_TRUE(arrayref.slice(2, 1).equals(values));
  }
  {
    value_type values[] = {1, 2};
    EXPECT_TRUE(arrayref.slice(0, 2).equals(values));
  }
  {
    value_type values[] = {2, 3};
    EXPECT_TRUE(arrayref.slice(1, 2).equals(values));
  }
  {
    value_type values[] = {1, 2, 3};
    EXPECT_TRUE(arrayref.slice(0, 3).equals(values));
  }
}

TYPED_TEST(LlvmLibcArrayRefTest, DropFront, Types) {
  using value_type = typename ParamType::value_type;
  value_type values[] = {1, 2, 3};
  ParamType arrayref(values);
  {
    value_type values[] = {1, 2, 3};
    EXPECT_TRUE(arrayref.drop_front(0).equals(values));
  }
  {
    value_type values[] = {2, 3};
    EXPECT_TRUE(arrayref.drop_front(1).equals(values));
  }
  {
    value_type values[] = {3};
    EXPECT_TRUE(arrayref.drop_front(2).equals(values));
  }
  { EXPECT_TRUE(arrayref.drop_front(3).empty()); }
}

TYPED_TEST(LlvmLibcArrayRefTest, DropBack, Types) {
  using value_type = typename ParamType::value_type;
  value_type values[] = {1, 2, 3};
  ParamType arrayref(values);
  {
    value_type values[] = {1, 2, 3};
    EXPECT_TRUE(arrayref.drop_back(0).equals(values));
  }
  {
    value_type values[] = {1, 2};
    EXPECT_TRUE(arrayref.drop_back(1).equals(values));
  }
  {
    value_type values[] = {1};
    EXPECT_TRUE(arrayref.drop_back(2).equals(values));
  }
  { EXPECT_TRUE(arrayref.drop_back(3).empty()); }
}

TYPED_TEST(LlvmLibcArrayRefTest, TakeFront, Types) {
  using value_type = typename ParamType::value_type;
  value_type values[] = {1, 2, 3};
  ParamType arrayref(values);
  { EXPECT_TRUE(arrayref.take_front(0).empty()); }
  {
    value_type values[] = {1};
    EXPECT_TRUE(arrayref.take_front(1).equals(values));
  }
  {
    value_type values[] = {1, 2};
    EXPECT_TRUE(arrayref.take_front(2).equals(values));
  }
  {
    value_type values[] = {1, 2, 3};
    EXPECT_TRUE(arrayref.take_front(3).equals(values));
  }
}

TYPED_TEST(LlvmLibcArrayRefTest, TakeBack, Types) {
  using value_type = typename ParamType::value_type;
  value_type values[] = {1, 2, 3};
  ParamType arrayref(values);
  { EXPECT_TRUE(arrayref.take_back(0).empty()); }
  {
    value_type values[] = {3};
    EXPECT_TRUE(arrayref.take_back(1).equals(values));
  }
  {
    value_type values[] = {2, 3};
    EXPECT_TRUE(arrayref.take_back(2).equals(values));
  }
  {
    value_type values[] = {1, 2, 3};
    EXPECT_TRUE(arrayref.take_back(3).equals(values));
  }
}

TEST(LlvmLibcArrayRefTest, ConstructFromVoidPtr) {
  unsigned data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  void *ptr = data;
  const void *const_ptr = data;
  ArrayRef<unsigned> ref(const_ptr, sizeof(data));
  MutableArrayRef<unsigned> mutable_ref(ptr, sizeof(data));
  ASSERT_EQ(ref.size(), sizeof(data) / sizeof(unsigned));
  ASSERT_EQ(mutable_ref.size(), sizeof(data) / sizeof(unsigned));

  unsigned val = 123;
  for (size_t i = 0; i < sizeof(data) / sizeof(unsigned); ++i)
    mutable_ref[i] = val;

  for (size_t i = 0; i < sizeof(data) / sizeof(unsigned); ++i)
    ASSERT_EQ(ref[i], val);
}

} // namespace cpp
} // namespace __llvm_libc
