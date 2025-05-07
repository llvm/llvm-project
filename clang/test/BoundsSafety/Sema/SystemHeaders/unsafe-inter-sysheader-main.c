
#include <unsafe-inter-sysheader-sys.h>

// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include -x objective-c -fexperimental-bounds-safety-objc
// expected-no-diagnostics
//
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc

void func(int * __unsafe_indexable unsafe, int * __single safe) {
  funcInSDK(unsafe);
  funcInSDK2(unsafe);
  funcInSDK3(safe);
  funcInSDK4(safe);
}

