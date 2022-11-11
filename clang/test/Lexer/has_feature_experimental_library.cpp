// RUN: %clang_cc1 -E -fexperimental-library %s -o - | FileCheck --check-prefix=CHECK-EXPERIMENTAL %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-EXPERIMENTAL %s

#if __has_feature(experimental_library)
int has_experimental_library();
#else
int has_no_experimental_library();
#endif
// CHECK-EXPERIMENTAL: int has_experimental_library();
// CHECK-NO-EXPERIMENTAL: int has_no_experimental_library();
