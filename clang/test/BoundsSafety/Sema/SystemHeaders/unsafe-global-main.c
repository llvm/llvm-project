
#include <unsafe-global-sys.h>

// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include -x objective-c -fexperimental-bounds-safety-objc
// expected-no-diagnostics
//
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc

void func(int * __unsafe_indexable unsafe, int * __terminated_by(2) term) {
  funcInSDK(unsafe);
  funcInSDK2(term);
  funcInSDK3(unsafe);
  funcInSDK4(term);
}
