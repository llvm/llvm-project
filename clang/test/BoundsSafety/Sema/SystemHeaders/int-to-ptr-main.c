
#include <int-to-ptr-sys.h>

// RUN: %clang_cc1 -fbounds-safety %s -verify=both -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify=both -I %S/include -x objective-c -fexperimental-bounds-safety-objc
//
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict,both -fno-bounds-safety-relaxed-system-headers -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify=strict,both -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc

int * func(intptr_t y) {
  // both-error@+1{{returning 'int *' from a function with incompatible result type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  return funcSDK(y);
}
