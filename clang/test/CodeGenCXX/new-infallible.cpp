// RUN: %clang_cc1 -emit-llvm -triple x86_64-linux-gnu -fnew-infallible -o - %s | FileCheck %s

// CHECK: call noalias noundef nonnull i8* @_Znwm(i64 noundef 4)

// CHECK: ; Function Attrs: nobuiltin nounwind allocsize(0)
// CHECK-NEXT: declare noundef nonnull i8* @_Znwm(i64 noundef)
int *new_infallible = new int;
