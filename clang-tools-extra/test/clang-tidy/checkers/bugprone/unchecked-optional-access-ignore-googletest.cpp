// RUN: %check_clang_tidy -check-suffix=IGNORE-TESTS %s \
// RUN:   bugprone-unchecked-optional-access %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:            {bugprone-unchecked-optional-access.IgnoreTestTUs: true}}" \
// RUN:   -- -I %S/Inputs/unchecked-optional-access

// RUN: %check_clang_tidy -check-suffix=DEFAULT %s \
// RUN:   bugprone-unchecked-optional-access %t -- \
// RUN:   -- -I %S/Inputs/unchecked-optional-access

#include "absl/types/optional.h"
#include "gtest/gtest.h"

// When IgnoreTestTUs is set, all of the warnings in a test TU are suppressed,
// even the `unchecked_value_access` and `assert_true_incorrect` cases.

// CHECK-MESSAGES-IGNORE-TESTS: 2 warnings generated
// CHECK-MESSAGES-DEFAULT: 6 warnings generated

// False negative from suppressing in test TU.
void unchecked_value_access(const absl::optional<int> opt) {
  opt.value();
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

// False negative from suppressing in test TU.
void assert_true_incorrect(const absl::optional<int> opt) {
  ASSERT_TRUE(!opt.has_value());
  opt.value();
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

// False positive, unless we suppress in test TU.
void assert_true_check_has_value(const absl::optional<int> opt) {
  ASSERT_TRUE(opt.has_value());
  *opt;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:4: warning: unchecked access to optional value
}

// False positive, unless we suppress (one of many other ways to check)
void assert_true_check_operator_bool_not_false(const absl::optional<int> opt) {
  ASSERT_FALSE(!opt);
  opt.value();
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

// CHECK-MESSAGES-IGNORE-TESTS: Suppressed 2 warnings
// CHECK-MESSAGES-DEFAULT: Suppressed 2 warnings
