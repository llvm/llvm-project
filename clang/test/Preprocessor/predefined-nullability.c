// RUN: %clang_cc1 %s -E -dM -triple i386-apple-darwin10 -o - | FileCheck %s --check-prefix=CHECK-DARWIN

// RUN: %clang_cc1 %s -E -dM -triple x86_64-unknown-linux -o - | FileCheck %s --check-prefix=CHECK-NONDARWIN


// CHECK-DARWIN: #define __nonnull _Nonnull
// CHECK-DARWIN: #define __null_unspecified _Null_unspecified
// CHECK-DARWIN: #define __nullable _Nullable

// CHECK-NONDARWIN-NOT: __nonnull
// CHECK-NONDARWIN: #define __clang__
// CHECK-NONDARWIN-NOT: __nonnull
