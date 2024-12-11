
#include <ptrcheck.h>

// ptrcheck macros should not affect code when compiled without -fbounds-safety
// RUN: %clang_cc1 -fsyntax-only -pedantic -D TEST_UNSAFE %s -verify=expected

// RUN: %clang_cc1 -fsyntax-only -pedantic -fbounds-safety %s -verify=expected
// RUN: %clang_cc1 -fsyntax-only -pedantic -fbounds-safety -D TEST_UNSAFE %s -verify=unsafe

// ptcheck macros should not affect code when compiled with -fexperimental-bounds-safety-attributes

// RUN: %clang_cc1 -fsyntax-only -pedantic -fexperimental-bounds-safety-attributes -x c %s -verify=expected
// RUN: %clang_cc1 -fsyntax-only -pedantic -fexperimental-bounds-safety-attributes -x c -D TEST_UNSAFE %s -verify=expected

// RUN: %clang_cc1 -fsyntax-only -pedantic -fexperimental-bounds-safety-attributes -x c++ %s -verify=expected
// RUN: %clang_cc1 -fsyntax-only -pedantic -fexperimental-bounds-safety-attributes -x c++ -D TEST_UNSAFE %s -verify=expected

// RUN: %clang_cc1 -fsyntax-only -pedantic -fexperimental-bounds-safety-attributes -x objective-c %s -verify=expected
// RUN: %clang_cc1 -fsyntax-only -pedantic -fexperimental-bounds-safety-attributes -x objective-c -D TEST_UNSAFE %s -verify=expected

// RUN: %clang_cc1 -fsyntax-only -pedantic -fexperimental-bounds-safety-attributes -x objective-c++ %s -verify=expected
// RUN: %clang_cc1 -fsyntax-only -pedantic -fexperimental-bounds-safety-attributes -x objective-c++ -D TEST_UNSAFE %s -verify=expected

// expected-no-diagnostics

int * __ptrcheck_unavailable unsafe_func(void); // unsafe-note{{'unsafe_func' has been explicitly marked unavailable here}}
int * __ptrcheck_unavailable_r(safe_func) unsafe_func2(void); // unsafe-note{{'unsafe_func2' has been explicitly marked unavailable here}}
void safe_func(int *__counted_by(*out_size) * out_arr, int *out_size);

#ifdef TEST_UNSAFE
void test1(void) {
    int *buf = unsafe_func(); // unsafe-error{{'unsafe_func' is unavailable: unavailable with -fbounds-safety}}
    buf = unsafe_func2(); // unsafe-error{{'unsafe_func2' is unavailable: unavailable with -fbounds-safety. Use safe_func instead.}}
}
#endif

void test2(void) {
    int len;
    int * __counted_by(len) buf;
    safe_func(&buf, &len);
}
