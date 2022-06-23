// RUN: %check_clang_tidy -check-suffix=DEFAULT %s \
// RUN: cppcoreguidelines-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: [{key: cppcoreguidelines-narrowing-conversions.WarnOnIntegerNarrowingConversion, value: true}]}'

// RUN: %check_clang_tidy -check-suffix=DISABLED %s \
// RUN: cppcoreguidelines-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: [{key: cppcoreguidelines-narrowing-conversions.WarnOnIntegerNarrowingConversion, value: false}]}'

void foo(unsigned long long value) {
  int a = value;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:11: warning: narrowing conversion from 'unsigned long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  // DISABLED: No warning for integer narrowing conversions when WarnOnIntegerNarrowingConversion = false.
  long long b = value;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:17: warning: narrowing conversion from 'unsigned long long' to signed type 'long long' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  // DISABLED: No warning for integer narrowing conversions when WarnOnIntegerNarrowingConversion = false.
}

void casting_float_to_bool_is_still_operational_when_integer_narrowing_is_disabled(float f) {
  if (f) {
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'bool' [cppcoreguidelines-narrowing-conversions]
    // CHECK-MESSAGES-DISABLED: :[[@LINE-2]]:7: warning: narrowing conversion from 'float' to 'bool' [cppcoreguidelines-narrowing-conversions]
  }
}
