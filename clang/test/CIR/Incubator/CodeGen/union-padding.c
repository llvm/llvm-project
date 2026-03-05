// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

typedef union {    
   short f0;
   signed   f1 : 11;
   unsigned f2 : 2;
   signed   f3 : 5;
} U;

static U g1[2] = {{65534UL}, {65534UL}};
static short *g2[1] = {&g1[1].f0};
static short **g3 = &g2[0];

short use() {
  U u;
  return **g3;
}
// CHECK:       !rec_U = !cir.record<union "U" padded {!s16i, !u16i, !u8i, !u8i, !cir.array<!u8i x 2>}>
// CHECK:       !rec_anon_struct = !cir.record<struct  {!s16i, !cir.array<!u8i x 2>}>

// CHECK:       @g3 = #cir.global_view<@g2> : !cir.ptr<!cir.ptr<!s16i>>
// CHECK:       @g2 = #cir.const_array<[#cir.global_view<@g1, [1]> : !cir.ptr<!s16i>]> : !cir.array<!cir.ptr<!s16i> x 1>

// CHECK:       @g1 = 
// CHECK-SAME:    #cir.const_array<[
// CHECK-SAME:      #cir.const_record<{#cir.int<-2> : !s16i, 
// CHECK-SAME:      #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 2>}> : !rec_anon_struct, 
// CHECK-SAME:      #cir.const_record<{#cir.int<-2> : !s16i,
// CHECK-SAME:      #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 2>}> : !rec_anon_struct
// CHECK-SAME:    ]> : !cir.array<!rec_anon_struct x 2>


