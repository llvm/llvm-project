

// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

unsigned global_data_const __unsafe_late_const;
void *global_data_const2 __unsafe_late_const;

// expected-error@+2{{'__unsafe_late_const' attribute only applies to global variables}}
// expected-error@+2{{'__unsafe_late_const' attribute only applies to global variables}}
void __unsafe_late_const
test(int param __unsafe_late_const) {
  static long static_local __unsafe_late_const;
  // expected-error@+1{{'__unsafe_late_const' attribute only applies to global variables}}
  int local __unsafe_late_const;
}
