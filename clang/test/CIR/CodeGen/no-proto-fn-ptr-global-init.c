// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -w -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Because LLVM IR uses opaque pointers, nothing in this test would be worth
// checking there, so LLVM and OGCG checks are omitted.

// Exercise replacing a no-prototype declaration with a prototyped definition
// when a global struct initializer already took the address of that function.
// The initializer must use the struct field's function-pointer type (not the
// FuncOp's no-prototype type) so CIR stays valid after the replacement.

// CHECK: !{{[^ ]+}} = !cir.record<struct "S1" {!cir.ptr<!cir.func<()>>, !s32i, !s32i}>
// CHECK: !{{[^ ]+}} = !cir.record<struct "S2" {!cir.ptr<!cir.func<(...)>>, !s32i, !s32i}>
// CHECK: !{{[^ ]+}} = !cir.record<struct "S3" {!cir.ptr<!cir.func<(...)>>, !s32i, !s32i}>
// CHECK: !{{[^ ]+}} = !cir.record<struct "S4" {!cir.ptr<!cir.func<(...)>>, !s32i, !s32i}>

struct S1 {
  void (*fn_ptr)(void);
  int a;
  int b;
};

void f1();

struct S1 s1 = {f1, 0, 1};

void f1(void) {}

// CHECK: cir.global {{.*}} @s1 = #cir.const_record<{#cir.global_view<@f1> : !cir.ptr<!cir.func<()>>, #cir.int<0> : !s32i, #cir.int<1> : !s32i}>

struct S2 {
  void (*fn_ptr)();
  int a;
  int b;
};

void f2();

struct S2 s2 = {f2, 0, 1};

void f2(void) {}

// CHECK: cir.global {{.*}} @s2 = #cir.const_record<{#cir.global_view<@f2> : !cir.ptr<!cir.func<(...)>>, #cir.int<0> : !s32i, #cir.int<1> : !s32i}>

struct S3 {
  void (*fn_ptr)();
  int a;
  int b;
};

void f3();

struct S3 s3 = {f3, 0, 1};

void f3(int x) {}

// CHECK: cir.global {{.*}} @s3 = #cir.const_record<{#cir.global_view<@f3> : !cir.ptr<!cir.func<(...)>>, #cir.int<0> : !s32i, #cir.int<1> : !s32i}>

struct S4 {
  void (*fn_ptr)();
  int a;
  int b;
};

void f4(int x);

// In this case we are initializing with a fully prototyped function. The
// initializer should still match the struct field's function-pointer type.
struct S4 s4 = {f4, 0, 1};

// CHECK: cir.global {{.*}} @s4 = #cir.const_record<{#cir.global_view<@f4> : !cir.ptr<!cir.func<(...)>>, #cir.int<0> : !s32i, #cir.int<1> : !s32i}>
