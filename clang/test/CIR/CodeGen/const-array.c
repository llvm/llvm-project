// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

void bar() {
  const int arr[1] = {1};
}

// CHECK: cir.global "private" constant internal @bar.arr = #cir.const_array<[#cir.int<1> : !s32i]> : !cir.array<!s32i x 1> {alignment = 4 : i64}
// CHECK: cir.func no_proto @bar()
// CHECK:   {{.*}} = cir.get_global @bar.arr : cir.ptr <!cir.array<!s32i x 1>>

void foo() {
  int a[10] = {1};
}

// CHECK: cir.func {{.*@foo}}
// CHECK:   %0 = cir.alloca !cir.array<!s32i x 10>, cir.ptr <!cir.array<!s32i x 10>>, ["a"] {alignment = 16 : i64}
// CHECK:   %1 = cir.const(#cir.const_array<[#cir.int<1> : !s32i], trailing_zeros> : !cir.array<!s32i x 10>) : !cir.array<!s32i x 10>
// CHECK:   cir.store %1, %0 : !cir.array<!s32i x 10>, cir.ptr <!cir.array<!s32i x 10>>
