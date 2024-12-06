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
// even the unchecked_value_access case.

// CHECK-MESSAGES-IGNORE-TESTS: 1 warning generated
// CHECK-MESSAGES-DEFAULT: 2 warnings generated

void unchecked_value_access(const absl::optional<int> &opt) {
  opt.value();
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

void assert_true_check_operator_access(const absl::optional<int> &opt) {
  ASSERT_TRUE(opt.has_value());
  *opt;
}

void assert_true_check_value_access(const absl::optional<int> &opt) {
  ASSERT_TRUE(opt.has_value());
  opt.value();
}

// CHECK-MESSAGES-IGNORE-TESTS: Suppressed 1 warnings
// CHECK-MESSAGES-DEFAULT: Suppressed 1 warnings
