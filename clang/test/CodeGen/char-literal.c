// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-C %s
// RUN: %clang_cc1 -x c++ -std=c++11 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-CPP0X %s

#include <stddef.h>

int main() {
  // CHECK-C: store i8 97
  // CHECK-CPP0X: store i8 97
  char a = 'a';

  // Should truncate value (equal to last character).
  // CHECK-C: store i8 98
  // CHECK-CPP0X: store i8 98
  char b = 'ab';

  // Should get concatenated characters
  // CHECK-C: store i32 24930
  // CHECK-CPP0X: store i32 24930
  int b1 = 'ab';

  // Should get concatenated characters
  // CHECK-C: store i32 808464432
  // CHECK-CPP0X: store i32 808464432
  int b2 = '0000';

  // Should get truncated value (last four characters concatenated)
  // CHECK-C: store i32 1919512167
  // CHECK-CPP0X: store i32 1919512167
  int b3 = 'somesillylongstring';

  // CHECK-C: store i32 97
  // CHECK-CPP0X: store i32 97
  wchar_t wa = L'a';

#if __cplusplus >= 201103L
  // CHECK-CPP0X: store i16 97
  char16_t ua = u'a';

  // CHECK-CPP0X: store i32 97
  char32_t Ua = U'a';

  // CHECK-CPP0X: store i16 1047
  char16_t ua1 = u'З';
  // CHECK-CPP0X: store i16 12538
  char16_t ua2 = u'ヺ';
  // CHECK-CPP0X: store i16 -27177
  char16_t ua3 = u'闗';

  // CHECK-CPP0X: store i32 181
  char32_t Ua1 = U'µ';
  // CHECK-CPP0X: store i32 38359
  char32_t Ua2 = U'闗';
  // CHECK-CPP0X: store i32 128128
  char32_t Ua3 = U'💀';

#endif

  // CHECK-C: store i32 61451
  // CHECK-CPP0X: store i32 61451
  wchar_t wc = L'\uF00B';

#if __cplusplus >= 201103L
  // -4085 == 0xf00b
  // CHECK-CPP0X: store i16 -4085
  char16_t uc = u'\uF00B';

  // CHECK-CPP0X: store i32 61451
  char32_t Uc = U'\uF00B';
#endif

  // CHECK-C: store i32 1110027
  // CHECK-CPP0X: store i32 1110027
  wchar_t wd = L'\U0010F00B';

#if __cplusplus >= 201103L
  // CHECK-CPP0X: store i32 1110027
  char32_t Ud = U'\U0010F00B';
#endif

}
