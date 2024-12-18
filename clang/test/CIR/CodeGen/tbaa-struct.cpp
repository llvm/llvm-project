// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// CIR: #tbaa[[tbaa_NYI:.*]] = #cir.tbaa
// CIR: #tbaa[[INT:.*]] = #cir.tbaa_scalar<type = !u32i>
// CIR: #tbaa[[INT_PTR:.*]] = #cir.tbaa_scalar<type = !cir.ptr<!u32i>>
// CIR: #tbaa[[StructA_PTR:.*]] = #cir.tbaa_scalar<type = !cir.ptr<!ty_StructA>>

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef struct
{
   uint16_t f16;
   uint32_t f32;
   uint16_t f16_2;
   uint32_t f32_2;
} StructA;

uint32_t g(uint32_t *s, StructA *A) {
  // CIR-LABEL: cir.func @_Z1g
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[INT]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: %[[pointer_to_StructA:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!ty_StructA>>, !cir.ptr<!ty_StructA> tbaa(#tbaa[[StructA_PTR]])
  // CIR: %[[A_f32:.*]] = cir.get_member %[[pointer_to_StructA]][1] {name = "f32"} : !cir.ptr<!ty_StructA> -> !cir.ptr<!u32i>
  // CIR: cir.store %[[UINT_4]], %[[A_f32]] : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[tbaa_NYI]])

  *s = 1;
  A->f32 = 4;
  return *s;
}
