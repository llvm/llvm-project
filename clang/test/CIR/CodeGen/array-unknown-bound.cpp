// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

extern int table[];
// CHECK: cir.global external @table = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>

int *table_ptr = table;
// CHECK: cir.global external @table_ptr = #cir.global_view<@table> : !cir.ptr<!s32i>

int test() { return table[1]; }
//      CHECK: cir.func @_Z4testv()
// CHECK-NEXT:    %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:    %1 = cir.get_global @table : cir.ptr <!cir.array<!s32i x 3>>

int table[3] {1, 2, 3};
