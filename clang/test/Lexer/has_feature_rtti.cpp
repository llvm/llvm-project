// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-RTTI %s
// RUN: %clang_cc1 -E -fno-rtti %s -o - | FileCheck --check-prefix=CHECK-NO-RTTI %s

#if __has_feature(cxx_rtti)
int foo();
#else
int bar();
#endif

// CHECK-RTTI: foo
// CHECK-NO-RTTI: bar
