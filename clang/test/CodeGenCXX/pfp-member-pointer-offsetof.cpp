// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -fexperimental-allow-pointer-field-protection-attr -fexperimental-pointer-field-protection-abi -o - %s | FileCheck %s

// CHECK: @__pfp_ds__ZTS1S.ptr1 = hidden alias i8, inttoptr (i64 3573751839 to ptr)
// CHECK: @__pfp_ds__ZTS1S.ptr2 = hidden alias i8, inttoptr (i64 3573751839 to ptr)

struct [[clang::pointer_field_protection]] S {
  int *ptr1;
  int *ptr2;
};

void f() {
  &S::ptr1;
  __builtin_offsetof(S, ptr2);
}
