// RUN: %check_clang_tidy %s cppcoreguidelines-narrowing-conversions %t \
// RUN: -config="{CheckOptions: { \
// RUN:   cppcoreguidelines-narrowing-conversions.PedanticMode: true}}" \
// RUN: -- -target x86_64-unknown-linux -fsigned-char

namespace floats {

void triggers_wrong_constant_type_warning(double d) {
  int i = 0.0;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: constant value should be of type of type 'int' instead of 'double' [cppcoreguidelines-narrowing-conversions]
  i += 2.0;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: constant value should be of type of type 'int' instead of 'double' [cppcoreguidelines-narrowing-conversions]
  i += 2.0f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: constant value should be of type of type 'int' instead of 'float' [cppcoreguidelines-narrowing-conversions]
}

void triggers_narrowing_warning_when_overflowing() {
  unsigned short us = 65537.0;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: narrowing conversion from constant 'double' to 'unsigned short' [cppcoreguidelines-narrowing-conversions]
}

} // namespace floats
