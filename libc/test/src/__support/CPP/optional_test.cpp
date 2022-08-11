//===-- Unittests for Optional --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/optional.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::cpp::nullopt;
using __llvm_libc::cpp::optional;

// This class has three properties for testing:
// 1) No default constructor.
// 2) A non-trivial destructor with an observable side-effect.
// 3) Functions that can be called explicitly.
class Contrived {
  int *_a;

public:
  Contrived(int *a) : _a(a) {}
  ~Contrived() { (*_a)++; }

  int get_a() { return *_a; }
  void inc_a() { (*_a)++; }
};

TEST(LlvmLibcOptionalTest, Tests) {
  optional<int> Trivial1(12);
  ASSERT_TRUE(Trivial1.has_value());
  ASSERT_EQ(Trivial1.value(), 12);
  ASSERT_EQ(*Trivial1, 12);
  Trivial1.reset();
  ASSERT_FALSE(Trivial1.has_value());

  optional<int> Trivial2(12);
  ASSERT_TRUE(Trivial2.has_value());
  Trivial2 = nullopt;
  ASSERT_FALSE(Trivial2.has_value());

  // For this test case, the destructor increments the pointed-to value.
  int holding = 1;
  optional<Contrived> Complicated(&holding);
  // Destructor was run once as part of copying the object.
  ASSERT_EQ(holding, 2);
  // Destructor was run a second time as part of destruction.
  Complicated.reset();
  ASSERT_EQ(holding, 3);
  // Destructor was not run a third time as the object is already destroyed.
  Complicated.reset();
  ASSERT_EQ(holding, 3);

  // Test that assigning an optional to another works when set
  optional<int> Trivial3(12);
  optional<int> Trivial4 = Trivial3;
  ASSERT_TRUE(Trivial4.has_value());
  ASSERT_EQ(Trivial4.value(), 12);

  // Test that assigning an option to another works when unset
  optional<int> Trivial5;
  ASSERT_FALSE(Trivial5.has_value());
  optional<int> Trivial6 = Trivial5;
  ASSERT_FALSE(Trivial6.has_value());

  // Test operator->
  int arrow_num = 5;
  optional<Contrived> arrow_test(&arrow_num);
  ASSERT_TRUE(arrow_test.has_value());
  ASSERT_EQ(arrow_test->get_a(), arrow_num);
  arrow_num = 10;
  ASSERT_EQ(arrow_test->get_a(), arrow_num);
  arrow_test->inc_a();
  ASSERT_EQ(arrow_test->get_a(), arrow_num);
  ASSERT_EQ(arrow_num, 11);
  arrow_test.reset();
}
