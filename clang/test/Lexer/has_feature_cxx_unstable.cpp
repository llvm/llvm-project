// RUN: %clang_cc1 -funstable -E %s -o - | FileCheck --check-prefix=CHECK-UNSTABLE %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-UNSTABLE %s

#if __has_feature(cxx_unstable)
int has_cxx_unstable();
#else
int has_no_cxx_unstable();
#endif
// CHECK-UNSTABLE: int has_cxx_unstable();
// CHECK-NO-UNSTABLE: int has_no_cxx_unstable();
