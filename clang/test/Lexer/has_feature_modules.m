// RUN: %clang_cc1 -E -fmodules %s -o - | FileCheck --check-prefix=CHECK-HAS-OBJC-MODULES %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-OBJC-MODULES %s
// RUN: %clang_cc1 -E -x c -fmodules %s -o - | FileCheck --check-prefix=CHECK-NO-OBJC-MODULES %s

// RUN: %clang_cc1 -E -fmodules %s -o - | FileCheck --check-prefix=CHECK-HAS-MODULES %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-MODULES %s
// RUN: %clang_cc1 -E -x c -fmodules %s -o - | FileCheck --check-prefix=CHECK-HAS-MODULES %s

#if __has_feature(modules)
int has_modules();
#else
int no_modules();
#endif

// CHECK-HAS-MODULES: has_modules
// CHECK-NO-MODULES: no_modules

#if __has_feature(objc_modules)
int has_objc_modules();
#else
int no_objc_modules();
#endif

// CHECK-HAS-OBJC-MODULES: has_objc_modules
// CHECK-NO-OBJC-MODULES: no_objc_modules
