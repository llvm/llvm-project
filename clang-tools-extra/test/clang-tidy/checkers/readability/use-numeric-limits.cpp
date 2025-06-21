// RUN: %check_clang_tidy %s readability-use-numeric-limits %t
// CHECK-FIXES: #include <limits>

using int8_t = signed char;
using int16_t = short;
using int32_t = int;
using int64_t = long long;
using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;


void Invalid() {
  // CHECK-MESSAGES: :[[@LINE+2]]:14: warning: the constant '-128' is being utilized; consider using 'std::numeric_limits<int8_t>::min()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int8_t a = std::numeric_limits<int8_t>::min();
  int8_t a = -128;

  // CHECK-MESSAGES: :[[@LINE+2]]:14: warning: the constant '127' is being utilized; consider using 'std::numeric_limits<int8_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int8_t b = std::numeric_limits<int8_t>::max();
  int8_t b = +127;

  // CHECK-MESSAGES: :[[@LINE+2]]:14: warning: the constant '127' is being utilized; consider using 'std::numeric_limits<int8_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int8_t c = std::numeric_limits<int8_t>::max();
  int8_t c = 127;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '-32768' is being utilized; consider using 'std::numeric_limits<int16_t>::min()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int16_t d = std::numeric_limits<int16_t>::min();
  int16_t d = -32768;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '32767' is being utilized; consider using 'std::numeric_limits<int16_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int16_t e = std::numeric_limits<int16_t>::max();
  int16_t e = +32767;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '32767' is being utilized; consider using 'std::numeric_limits<int16_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int16_t f = std::numeric_limits<int16_t>::max();
  int16_t f = 32767;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '-2147483648' is being utilized; consider using 'std::numeric_limits<int32_t>::min()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int32_t g = std::numeric_limits<int32_t>::min();
  int32_t g = -2147483648;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '2147483647' is being utilized; consider using 'std::numeric_limits<int32_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int32_t h = std::numeric_limits<int32_t>::max();
  int32_t h = +2147483647;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '2147483647' is being utilized; consider using 'std::numeric_limits<int32_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int32_t i = std::numeric_limits<int32_t>::max();
  int32_t i = 2147483647;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '-9223372036854775808' is being utilized; consider using 'std::numeric_limits<int64_t>::min()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int64_t j = std::numeric_limits<int64_t>::min();
  int64_t j = -9223372036854775808;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '9223372036854775807' is being utilized; consider using 'std::numeric_limits<int64_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int64_t k = std::numeric_limits<int64_t>::max();
  int64_t k = +9223372036854775807;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '9223372036854775807' is being utilized; consider using 'std::numeric_limits<int64_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: int64_t l = std::numeric_limits<int64_t>::max();
  int64_t l = 9223372036854775807;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '255' is being utilized; consider using 'std::numeric_limits<uint8_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint8_t m = std::numeric_limits<uint8_t>::max();
  uint8_t m = 255;

  // CHECK-MESSAGES: :[[@LINE+2]]:15: warning: the constant '255' is being utilized; consider using 'std::numeric_limits<uint8_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint8_t n = std::numeric_limits<uint8_t>::max();
  uint8_t n = +255;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: the constant '65535' is being utilized; consider using 'std::numeric_limits<uint16_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint16_t o = std::numeric_limits<uint16_t>::max();
  uint16_t o = 65535;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: the constant '65535' is being utilized; consider using 'std::numeric_limits<uint16_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint16_t p = std::numeric_limits<uint16_t>::max();
  uint16_t p = +65535;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: the constant '4294967295' is being utilized; consider using 'std::numeric_limits<uint32_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint32_t q = std::numeric_limits<uint32_t>::max();
  uint32_t q = 4294967295;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: the constant '4294967295' is being utilized; consider using 'std::numeric_limits<uint32_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint32_t r = std::numeric_limits<uint32_t>::max();
  uint32_t r = +4294967295;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: the constant '18446744073709551615' is being utilized; consider using 'std::numeric_limits<uint64_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint64_t s = std::numeric_limits<uint64_t>::max();
  uint64_t s = 18446744073709551615;

  // CHECK-MESSAGES: :[[@LINE+2]]:16: warning: the constant '18446744073709551615' is being utilized; consider using 'std::numeric_limits<uint64_t>::max()' instead [readability-use-numeric-limits]
  // CHECK-FIXES: uint64_t t = std::numeric_limits<uint64_t>::max();
  uint64_t t = +18446744073709551615;
}

void Valid(){
  int16_t a = +128;

  int16_t b = -127;
}
