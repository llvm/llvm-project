
#include <unsafe-global-sys.h>

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -I %S/include | FileCheck --check-prefixes RELAXED %S/include/unsafe-global-sys.h
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck --check-prefixes RELAXED %S/include/unsafe-global-sys.h
//
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include | FileCheck --check-prefixes STRICT,MAINCHECK %S/include/unsafe-global-sys.h
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck --check-prefixes STRICT,MAINCHECK %S/include/unsafe-global-sys.h

void func(int * __unsafe_indexable unsafe, int * __terminated_by(2) term) {
  funcInSDK(unsafe);
  funcInSDK2(term);
  funcInSDK3(unsafe);
  funcInSDK4(term);
}
