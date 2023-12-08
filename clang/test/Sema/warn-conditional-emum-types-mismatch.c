// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wenum-compare-conditional %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wenum-compare-conditional %s

enum ro { A = 0x10 };
enum rw { B = 0xFF };
enum { C = 0x1A};

enum {
  STATUS_SUCCESS,
  STATUS_FAILURE,
  MAX_BASE_STATUS_CODE
};

enum ExtendedStatusCodes {
  STATUS_SOMETHING_INTERESTING = MAX_BASE_STATUS_CODE + 1000,
};


int get_flag(int cond) {
  return cond ? A : B; 
  #ifdef __cplusplus
  // expected-warning@-2 {{conditional expression between different enumeration types ('ro' and 'rw')}}
  #else 
  // expected-no-diagnostics
  #endif
}

// In the following cases we purposefully differ from GCC and dont warn because
// this code pattern is quite sensitive and we dont want to produce so many false positives.

int get_flag_anon_enum(int cond) {
  return cond ? A : C;
}

int foo(int c) {
  return c ? STATUS_SOMETHING_INTERESTING : STATUS_SUCCESS;
}
