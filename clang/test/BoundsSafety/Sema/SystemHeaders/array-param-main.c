
#include <array-param-sys.h>

// RUN: %clang_cc1 -fbounds-safety %s -I %S/include -verify=strict -fno-bounds-safety-relaxed-system-headers

// RUN: %clang_cc1 -fbounds-safety %s -I %S/include -verify
// expected-no-diagnostics

void foo(int * __counted_by(size) arr, int size) {
    funcInSDK(size, arr);
}


