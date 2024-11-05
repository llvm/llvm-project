// RUN: %check_clang_tidy %s bugprone-unchecked-optional-access %t -- -- -I %S/Inputs/unchecked-optional-access

#include "absl/types/optional.h"
#include "gtest/gtest.h"

// All of the warnings are suppressed once we detect that we are in a test TU
// even the unchecked_value_access case.

// CHECK-MESSAGE: 1 warning generated
// CHECK-MESSAGES: Suppressed 1 warnings

void unchecked_value_access(const absl::optional<int> &opt) {
  opt.value();
}

void assert_true_check_operator_access(const absl::optional<int> &opt) {
  ASSERT_TRUE(opt.has_value());
  *opt;
}

void assert_true_check_value_access(const absl::optional<int> &opt) {
  ASSERT_TRUE(opt.has_value());
  opt.value();
}
