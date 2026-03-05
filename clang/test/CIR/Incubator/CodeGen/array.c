// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Should implicitly zero-initialize global array elements.
struct S {
  int i;
} arr[3] = {{1}};
// CHECK: cir.global external @arr = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i}> : !rec_S, #cir.zero : !rec_S, #cir.zero : !rec_S]> : !cir.array<!rec_S x 3>

int a[4];
// CHECK: cir.global external @a = #cir.zero : !cir.array<!s32i x 4>

// Should create a pointer to a complete array.
int (*complete_ptr_a)[4] = &a;
// CHECK: cir.global external @complete_ptr_a = #cir.global_view<@a> : !cir.ptr<!cir.array<!s32i x 4>>

// Should create a pointer to an incomplete array.
int (*incomplete_ptr_a)[] = &a;
// CHECK: cir.global external @incomplete_ptr_a = #cir.global_view<@a> : !cir.ptr<!cir.array<!s32i x 0>>

// Should access incomplete array if external.
extern int foo[];
// CHECK: cir.global "private" external @foo : !cir.array<!s32i x 0>
void useFoo(int i) {
  foo[i] = 42;
}
// CHECK: @useFoo
// CHECK: %[[#V2:]] = cir.get_global @foo : !cir.ptr<!cir.array<!s32i x 0>>
// CHECK: %[[#V3:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!s32i>, !s32i
// CHECK: %[[#V4:]] = cir.get_element %[[#V2]][%[[#V3]]] : (!cir.ptr<!cir.array<!s32i x 0>>, !s32i) -> !cir.ptr<!s32i>
// CHECK: cir.store{{.*}} %{{.+}}, %[[#V4]] : !s32i, !cir.ptr<!s32i>
