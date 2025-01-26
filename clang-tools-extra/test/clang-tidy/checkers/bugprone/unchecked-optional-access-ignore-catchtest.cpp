// RUN: %check_clang_tidy -check-suffix=IGNORE-TESTS %s \
// RUN:   bugprone-unchecked-optional-access %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:            {bugprone-unchecked-optional-access.IgnoreTestTUs: true}}" \
// RUN:   -- -I %S/Inputs/unchecked-optional-access

// RUN: %check_clang_tidy -check-suffix=IGNORE-TESTS-CATCH-PREFIX %s \
// RUN:   --extra-arg=-DCATCH_CONFIG_PREFIX_ALL \
// RUN:   bugprone-unchecked-optional-access %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:            {bugprone-unchecked-optional-access.IgnoreTestTUs: true}}" \
// RUN:   -- -I %S/Inputs/unchecked-optional-access

// RUN: %check_clang_tidy -check-suffix=DEFAULT %s \
// RUN:   bugprone-unchecked-optional-access %t -- \
// RUN:   -- -I %S/Inputs/unchecked-optional-access

// RUN: %check_clang_tidy -check-suffix=DEFAULT-CATCH-PREFIX %s \
// RUN:   --extra-arg=-DCATCH_CONFIG_PREFIX_ALL \
// RUN:   bugprone-unchecked-optional-access %t -- \
// RUN:   -- -I %S/Inputs/unchecked-optional-access

#include "absl/types/optional.h"
#include "catch2/catch_test_macros.hpp"

// When IgnoreTestTUs is set, all of the warnings in a test TU are suppressed,
// even the "Unguarded access" and "Guarded incorrectly" cases.

// CHECK-MESSAGES-IGNORE-TESTS: 2 warnings generated
// CHECK-MESSAGES-IGNORE-TESTS-CATCH-PREFIX: 1 warning generated
// CHECK-MESSAGES-DEFAULT: 6 warnings generated
// CHECK-MESSAGES-DEFAULT-CATCH-PREFIX: 3 warnings generated

absl::optional<int> FunctionUnderTest();

#ifndef CATCH_CONFIG_PREFIX_ALL

TEST_CASE("FN -- Unguarded access", "[optional]") {
  auto opt = FunctionUnderTest();
  opt.value();
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

TEST_CASE("FN -- Guarded incorrectly assert false has_value()", "[optional]") {
  auto opt = FunctionUnderTest();
  REQUIRE_FALSE(!opt.has_value());
  opt.value();
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

TEST_CASE("FP -- Guarded assert true has_value()", "[optional]") {
  auto opt = FunctionUnderTest();
  REQUIRE(opt.has_value());
  *opt;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:4: warning: unchecked access to optional value
}

TEST_CASE("FP -- Guarded assert false not operator bool()", "[optional]") {
  auto opt = FunctionUnderTest();
  REQUIRE_FALSE(!opt);
  opt.value();
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

#else

CATCH_TEST_CASE("FN -- Unguarded access", "[optional]") {
  auto opt = FunctionUnderTest();
  opt.value();
  // CHECK-MESSAGES-DEFAULT-CATCH-PREFIX: :[[@LINE-1]]:3: warning: unchecked access to optional value
}

CATCH_TEST_CASE("FP -- Guarded assert true has_value()", "[optional]") {
  auto opt = FunctionUnderTest();
  CATCH_REQUIRE(opt.has_value());
  *opt;
  // CHECK-MESSAGES-DEFAULT-CATCH-PREFIX: :[[@LINE-1]]:4: warning: unchecked access to optional value
}

#endif

// CHECK-MESSAGES-IGNORE-TESTS: Suppressed 2 warnings
// CHECK-MESSAGES-IGNORE-TESTS-CATCH-PREFIX: Suppressed 1 warning
// CHECK-MESSAGES-DEFAULT: Suppressed 2 warnings
// CHECK-MESSAGES-DEFAULT-CATCH-PREFIX: Suppressed 1 warning
