
#include <struct-fields-sys.h>

// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include -x objective-c -fexperimental-bounds-safety-objc
// expected-no-diagnostics

// RUN: %clang_cc1 -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc

void func(int * a, int * b, struct bar in, struct foo in2) {
  funcInSDK1(a, *b);
  funcInSDK2(a, *b);
  funcInSDK3(a, *b);
  funcInSDK4(a, *b);
  funcInSDK5(a, *b);
  funcInSDK6(a, b);
  funcInSDK7(a, b);
  funcInSDK8(a, b);
  funcInSDK9(in);
  funcInSDK10(in);
  funcInSDK11(in2);
  funcInSDK12(in2);
}
