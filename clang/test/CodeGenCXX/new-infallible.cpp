// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu -fnew-infallible -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu -fno-new-infallible -fnew-infallible -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu -fno-new-infallible -o - %s | FileCheck %s --check-prefix=NO-NEW-INFALLIBLE
// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu -fnew-infallible -fno-new-infallible -o - %s | FileCheck %s --check-prefix=NO-NEW-INFALLIBLE

// CHECK: call noalias nonnull i8* @_Znwm(i64 4)

// CHECK: ; Function Attrs: nobuiltin nounwind allocsize(0)
// CHECK-NEXT: declare nonnull i8* @_Znwm(i64)

// NO-NEW-INFALLIBLE: call noalias nonnull i8* @_Znwm(i64 4)

// NO-NEW-INFALLIBLE: ; Function Attrs: nobuiltin allocsize(0)
// NO-NEW-INFALLIBLE-NEXT: declare nonnull i8* @_Znwm(i64)

int *new_infallible = new int;
