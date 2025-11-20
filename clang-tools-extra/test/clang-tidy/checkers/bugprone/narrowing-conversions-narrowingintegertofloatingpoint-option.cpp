// RUN: %check_clang_tidy -check-suffix=DEFAULT %s \
// RUN: bugprone-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: {bugprone-narrowing-conversions.WarnOnIntegerToFloatingPointNarrowingConversion: true}}'

// RUN: %check_clang_tidy -check-suffix=DISABLED %s \
// RUN: bugprone-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: {bugprone-narrowing-conversions.WarnOnIntegerToFloatingPointNarrowingConversion: false}}'

void foo(unsigned long long value) {
  double a = value;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:14: warning: narrowing conversion from 'unsigned long long' to 'double' [bugprone-narrowing-conversions]
  // DISABLED: No warning for integer to floating-point narrowing conversions when WarnOnIntegerToFloatingPointNarrowingConversion = false.
}

void floating_point_to_integer_is_still_not_ok(double f) {
  int a = f;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: narrowing conversion from 'double' to 'int' [bugprone-narrowing-conversions]
  // CHECK-MESSAGES-DISABLED: :[[@LINE-2]]:11: warning: narrowing conversion from 'double' to 'int' [bugprone-narrowing-conversions]
}
