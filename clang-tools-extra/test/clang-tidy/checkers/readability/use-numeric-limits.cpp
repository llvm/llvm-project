// RUN: %check_clang_tidy -std=c++11-or-later %s readability-use-numeric-limits %t
#include <stdint.h>

void constants() {
  // CHECK-MESSAGES: :[[@LINE+2]]:14: warning: The constant -128 is being utilized. Consider using std::numeric_limits<int8_t>::min() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int8_t _ = std::numeric_limits<int8_t>::min();
  int8_t _ = -128;

  // CHECK-MESSAGES: :[[@LINE+2]]:14: warning: The constant 127 is being utilized. Consider using std::numeric_limits<int8_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int8_t _ = std::numeric_limits<int8_t>::max();
  int8_t _ = +127;

  // CHECK-MESSAGES: :[[@LINE+2]]:14: warning: The constant 127 is being utilized. Consider using std::numeric_limits<int8_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int8_t _ = std::numeric_limits<int8_t>::max();
  int8_t _ = 127;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant -32768 is being utilized. Consider using std::numeric_limits<int16_t>::min() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int16_t _ = std::numeric_limits<int16_t>::min();
  int16_t _ = -32768;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant 32767 is being utilized. Consider using std::numeric_limits<int16_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int16_t _ = std::numeric_limits<int16_t>::max();
  int16_t _ = +32767;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant 32767 is being utilized. Consider using std::numeric_limits<int16_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int16_t _ = std::numeric_limits<int16_t>::max();
  int16_t _ = 32767;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant -2147483648 is being utilized. Consider using std::numeric_limits<int32_t>::min() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int32_t _ = std::numeric_limits<int32_t>::min();
  int32_t _ = -2147483648;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant 2147483647 is being utilized. Consider using std::numeric_limits<int32_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int32_t _ = std::numeric_limits<int32_t>::max();
  int32_t _ = +2147483647;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant 2147483647 is being utilized. Consider using std::numeric_limits<int32_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int32_t _ = std::numeric_limits<int32_t>::max();
  int32_t _ = 2147483647;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant -9223372036854775808 is being utilized. Consider using std::numeric_limits<int64_t>::min() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int64_t _ = std::numeric_limits<int64_t>::min();
  int64_t _ = -9223372036854775808;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant 9223372036854775807 is being utilized. Consider using std::numeric_limits<int64_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int64_t _ = std::numeric_limits<int64_t>::max();
  int64_t _ = +9223372036854775807;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant 9223372036854775807 is being utilized. Consider using std::numeric_limits<int64_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: int64_t _ = std::numeric_limits<int64_t>::max();
  int64_t _ = 9223372036854775807;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant 255 is being utilized. Consider using std::numeric_limits<uint8_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint8_t _ = std::numeric_limits<uint8_t>::max();
  uint8_t _ = 255;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: The constant 255 is being utilized. Consider using std::numeric_limits<uint8_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint8_t _ = std::numeric_limits<uint8_t>::max();
  uint8_t _ = +255;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: The constant 65535 is being utilized. Consider using std::numeric_limits<uint16_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint16_t _ = std::numeric_limits<uint16_t>::max();
  uint16_t _ = 65535;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: The constant 65535 is being utilized. Consider using std::numeric_limits<uint16_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint16_t _ = std::numeric_limits<uint16_t>::max();
  uint16_t _ = +65535;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: The constant 4294967295 is being utilized. Consider using std::numeric_limits<uint32_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint32_t _ = std::numeric_limits<uint32_t>::max();
  uint32_t _ = 4294967295;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: The constant 4294967295 is being utilized. Consider using std::numeric_limits<uint32_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint32_t _ = std::numeric_limits<uint32_t>::max();
  uint32_t _ = +4294967295;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: The constant 18446744073709551615 is being utilized. Consider using std::numeric_limits<uint64_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint64_t _ = std::numeric_limits<uint64_t>::max();
  uint64_t _ = 18446744073709551615;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: The constant 18446744073709551615 is being utilized. Consider using std::numeric_limits<uint64_t>::max() instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint64_t _ = std::numeric_limits<uint64_t>::max();
  uint64_t _ = +18446744073709551615;
}
