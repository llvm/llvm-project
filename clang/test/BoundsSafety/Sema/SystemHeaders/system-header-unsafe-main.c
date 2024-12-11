
#include <system-header-unsafe-sys.h>

// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include -x objective-c -fbounds-attributes-objc-experimental
// expected-no-diagnostics

// RUN: %clang_cc1 -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fbounds-attributes-objc-experimental

void func(char * __unsafe_indexable ptr, char * __bidi_indexable bidi) {
  funcInSDK(ptr, bidi);
}
