
#include <typedefs-sys.h>

// RUN: %clang_cc1 -fbounds-safety %s -verify=both -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify=both -I %S/include -x objective-c -fexperimental-bounds-safety-objc
//
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict,both -fno-bounds-safety-relaxed-system-headers -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict,both -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc

void func(const char * a, int * b) {
  funcInSDK1(a);
  funcInSDK2(a);
  funcInSDK3(b);
  funcInSDK4(b);
}

