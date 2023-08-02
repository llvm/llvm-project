// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct Bar {
  int a;
  char b;
} bar;

struct Foo {
  int a;
  char b;
  struct Bar z;
};

void baz(void) {
  struct Bar b;
  struct Foo f;
}

//      CHECK: !ty_22struct2EBar22 = !cir.struct<"struct.Bar", !s32i, !s8i>
// CHECK-NEXT: !ty_22struct2EFoo22 = !cir.struct<"struct.Foo", !s32i, !s8i, !ty_22struct2EBar22>
//  CHECK-DAG: module {{.*}} {
// CHECK-NEXT:   cir.func @baz()
// CHECK-NEXT:     %0 = cir.alloca !ty_22struct2EBar22, cir.ptr <!ty_22struct2EBar22>, ["b"] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.alloca !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>, ["f"] {alignment = 4 : i64}
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }

void shouldConstInitStructs(void) {
// CHECK: cir.func @shouldConstInitStructs
  struct Foo f = {1, 2, {3, 4}};
  // CHECK: %[[#V0:]] = cir.alloca !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>, ["f"] {alignment = 4 : i64}
  // CHECK: %[[#V1:]] = cir.const(#cir.const_struct<{#cir.int<1> : !s32i, #cir.int<2> : !s8i, #cir.const_struct<{#cir.int<3> : !s32i, #cir.int<4> : !s8i}> : !ty_22struct2EBar22}> : !ty_22struct2EFoo22) : !ty_22struct2EFoo22
  // CHECK: cir.store %[[#V1]], %[[#V0]] : !ty_22struct2EFoo22, cir.ptr <!ty_22struct2EFoo22>
}

// Check if global structs are zero-initialized.
//      CHECK: cir.global external @bar = #cir.zero : !ty_22struct2EBar22
