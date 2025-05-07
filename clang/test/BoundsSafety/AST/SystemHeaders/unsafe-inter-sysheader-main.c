
#include <unsafe-inter-sysheader-sys.h>
//                                                                 
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -DSYSHEADER -I %S/include | FileCheck --check-prefixes RELAXED %S/include/unsafe-inter-sysheader-sys.h
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -DSYSHEADER -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck --check-prefixes RELAXED %S/include/unsafe-inter-sysheader-sys.h
//
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include | FileCheck --check-prefixes STRICT %S/include/unsafe-inter-sysheader-sys.h
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck --check-prefixes STRICT %S/include/unsafe-inter-sysheader-sys.h

void func(int * __unsafe_indexable unsafe, int * __single safe) {
  funcInSDK(unsafe);
  funcInSDK2(unsafe);
  funcInSDK3(safe);
  funcInSDK4(safe);
}

