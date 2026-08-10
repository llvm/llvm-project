// RUN: %clang -E -fexceptions %s -o - | FileCheck --check-prefix=CHECK-EXCEPTIONS %s
// RUN: %clang -E -fexceptions -fno-cxx-exceptions %s -o - | FileCheck --check-prefix=CHECK-NO-EXCEPTIONS %s
// RUN: %clang -E -fno-exceptions %s -o - | FileCheck --check-prefix=CHECK-NO-EXCEPTIONS %s

// RUN: %clang_cc1 -E -fcxx-exceptions %s -o - | FileCheck --check-prefix=CHECK-EXCEPTIONS %s
// RUN: %clang_cc1 -E -fobjc-exceptions %s -o - | FileCheck --check-prefix=CHECK-NO-EXCEPTIONS %s
// RUN: %clang_cc1 -E -fexceptions %s -o - | FileCheck --check-prefix=CHECK-NO-EXCEPTIONS %s
// RUN: %clang_cc1 -E %s -o - | FileCheck --check-prefix=CHECK-NO-EXCEPTIONS %s

#if __has_feature(cxx_exceptions)
int foo();
#else
int bar();
#endif

// CHECK-EXCEPTIONS: foo
// CHECK-NO-EXCEPTIONS: bar
