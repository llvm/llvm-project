// This is inspired from clang/test/CodeGen/tbaa.cpp, with both CIR and LLVM checks.
// g13 is not supported due to DiscreteBitFieldABI is NYI.
// see clang/lib/CIR/CodeGen/CIRRecordLayoutBuilder.cpp CIRRecordLowering::accumulateBitFields

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -disable-llvm-passes -no-struct-path-tbaa
// RUN: FileCheck --check-prefix=CHECK --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -disable-llvm-passes
// RUN: FileCheck --check-prefixes=PATH,OLD-PATH --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -disable-llvm-passes -relaxed-aliasing
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O0 -disable-llvm-passes
// RUN: FileCheck --check-prefix=NO-TBAA --input-file=%t.ll %s

// NO-TBAA-NOT: !tbaa
// CIR: #tbaa[[NYI:.*]] = #cir.tbaa
// CIR: #tbaa[[CHAR:.*]] = #cir.tbaa_omnipotent_char
// CIR: #tbaa[[INT:.*]] = #cir.tbaa_scalar<id = "int", type = !s32i>
// CIR: #tbaa[[SHORT:.*]] = #cir.tbaa_scalar<id = "short", type = !s16i>
// CIR: #tbaa[[STRUCT_six:.*]] = #cir.tbaa_struct<id = "_ZTS3six", members = {<#tbaa[[CHAR]], 0>, <#tbaa[[CHAR]], 4>, <#tbaa[[CHAR]], 5>}>
// CIR: #tbaa[[STRUCT_StructA:.*]] = #cir.tbaa_struct<id = "_ZTS7StructA", members = {<#tbaa[[SHORT]], 0>, <#tbaa[[INT]], 4>, <#tbaa[[SHORT]], 8>, <#tbaa[[INT]], 12>}>
// CIR: #tbaa[[STRUCT_StructS:.*]] = #cir.tbaa_struct<id = "_ZTS7StructS", members = {<#tbaa[[SHORT]], 0>, <#tbaa[[INT]], 4>}>
// CIR: #tbaa[[STRUCT_StructS2:.*]] = #cir.tbaa_struct<id = "_ZTS8StructS2", members = {<#tbaa[[SHORT]], 0>, <#tbaa[[INT]], 4>}>
// CIR: #tbaa[[TAG_six_b:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_six]], access = #tbaa[[CHAR]], offset = 4>
// CIR: #tbaa[[TAG_StructA_f32:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructA]], access = #tbaa[[INT]], offset = 4>
// CIR: #tbaa[[TAG_StructA_f16:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructA]], access = #tbaa[[SHORT]], offset = 0>
// CIR: #tbaa[[TAG_StructS_f32:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructS]], access = #tbaa[[INT]], offset = 4>
// CIR: #tbaa[[TAG_StructS_f16:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructS]], access = #tbaa[[SHORT]], offset = 0>
// CIR: #tbaa[[TAG_StructS2_f32:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructS2]], access = #tbaa[[INT]], offset = 4>
// CIR: #tbaa[[TAG_StructS2_f16:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructS2]], access = #tbaa[[SHORT]], offset = 0>
// CIR: #tbaa[[STRUCT_StructB:.*]] = #cir.tbaa_struct<id = "_ZTS7StructB", members = {<#tbaa[[SHORT]], 0>, <#tbaa[[STRUCT_StructA]], 4>, <#tbaa[[INT]], 20>}>
// CIR: #tbaa[[TAG_StructB_a_f32:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructB]], access = #tbaa[[INT]], offset = 8>
// CIR: #tbaa[[TAG_StructB_a_f16:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructB]], access = #tbaa[[SHORT]], offset = 4>
// CIR: #tbaa[[TAG_StructB_f32:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructB]], access = #tbaa[[INT]], offset = 20>
// CIR: #tbaa[[TAG_StructB_a_f32_2:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructB]], access = #tbaa[[INT]], offset = 16>
// CIR: #tbaa[[STRUCT_StructC:.*]] = #cir.tbaa_struct<id = "_ZTS7StructC", members = {<#tbaa[[SHORT]], 0>, <#tbaa[[STRUCT_StructB]], 4>, <#tbaa[[INT]], 28>}>
// CIR: #tbaa[[STRUCT_StructD:.*]] = #cir.tbaa_struct<id = "_ZTS7StructD", members = {<#tbaa[[SHORT]], 0>, <#tbaa[[STRUCT_StructB]], 4>, <#tbaa[[INT]], 28>, <#tbaa[[CHAR]], 32>}>
// CIR: #tbaa[[TAG_StructC_b_a_f32:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructC]], access = #tbaa[[INT]], offset = 12>
// CIR: #tbaa[[TAG_StructD_b_a_f32:.*]] = #cir.tbaa_tag<base = #tbaa[[STRUCT_StructD]], access = #tbaa[[INT]], offset = 12>


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
typedef struct
{
   uint16_t f16;
   StructA a;
   uint32_t f32;
} StructB;
typedef struct
{
   uint16_t f16;
   StructB b;
   uint32_t f32;
} StructC;
typedef struct
{
   uint16_t f16;
   StructB b;
   uint32_t f32;
   uint8_t f8;
} StructD;

typedef struct
{
   uint16_t f16;
   uint32_t f32;
} StructS;
typedef struct
{
   uint16_t f16;
   uint32_t f32;
} StructS2;

uint32_t g(uint32_t *s, StructA *A, uint64_t count) {
  // CIR-LABEL: cir.func @_Z1g
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[INT]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructA_f32]])


  // CHECK-LABEL: define{{.*}} i32 @_Z1g
  // CHECK: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32:!.*]]
  // CHECK: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH-LABEL: define{{.*}} i32 @_Z1g
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32:!.*]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_A_f32:!.*]]
  *s = 1;
  A->f32 = 4;
  return *s;
}

uint32_t g2(uint32_t *s, StructA *A, uint64_t count) {
  // CIR-LABEL: cir.func @_Z2g2
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[INT]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u16i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u16i, !cir.ptr<!u16i> tbaa(#tbaa[[TAG_StructA_f16]])

  // CHECK-LABEL: define{{.*}} i32 @_Z2g2
  // CHECK: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // CHECK: store i16 4, ptr %{{.*}}, align {{4|2}}, !tbaa [[TAG_i16:!.*]]
  // PATH-LABEL: define{{.*}} i32 @_Z2g2
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH: store i16 4, ptr %{{.*}}, align {{4|2}}, !tbaa [[TAG_A_f16:!.*]]
  *s = 1;
  A->f16 = 4;
  return *s;
}

uint32_t g3(StructA *A, StructB *B, uint64_t count) {
  // CIR-LABEL: cir.func @_Z2g3
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructA_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructB_a_f32]])

  // CHECK-LABEL: define{{.*}} i32 @_Z2g3
  // CHECK: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // CHECK: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH-LABEL: define{{.*}} i32 @_Z2g3
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_A_f32]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_B_a_f32:!.*]]
  A->f32 = 1;
  B->a.f32 = 4;
  return A->f32;
}

uint32_t g4(StructA *A, StructB *B, uint64_t count) {
  // CIR-LABEL: cir.func @_Z2g4
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructA_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u16i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u16i, !cir.ptr<!u16i> tbaa(#tbaa[[TAG_StructB_a_f16]])

  // LLVM-LABEL: define{{.*}} i32 @_Z2g4
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i16 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i16]]
  // PATH-LABEL: define{{.*}} i32 @_Z2g4
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_A_f32]]
  // PATH: store i16 4, ptr %{{.*}}, align {{4|2}}, !tbaa [[TAG_B_a_f16:!.*]]
  A->f32 = 1;
  B->a.f16 = 4;
  return A->f32;
}

uint32_t g5(StructA *A, StructB *B, uint64_t count) {
  // CIR-LABEL: cir.func @_Z2g5
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructA_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructB_f32]])

  // LLVM-LABEL: define{{.*}} i32 @_Z2g5
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH-LABEL: define{{.*}} i32 @_Z2g5
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_A_f32]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_B_f32:!.*]]
  A->f32 = 1;
  B->f32 = 4;
  return A->f32;
}

uint32_t g6(StructA *A, StructB *B, uint64_t count) {
  // CIR-LABEL: cir.func @_Z2g6
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructA_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructB_a_f32_2]])

  // LLVM-LABEL: define{{.*}} i32 @_Z2g6
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH-LABEL: define{{.*}} i32 @_Z2g6
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_A_f32]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_B_a_f32_2:!.*]]
  A->f32 = 1;
  B->a.f32_2 = 4;
  return A->f32;
}

uint32_t g7(StructA *A, StructS *S, uint64_t count) {
  // CIR-LABEL: cir.func @_Z2g7
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructA_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructS_f32]])

  // LLVM-LABEL: define{{.*}} i32 @_Z2g7
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH-LABEL: define{{.*}} i32 @_Z2g7
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_A_f32]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_S_f32:!.*]]
  A->f32 = 1;
  S->f32 = 4;
  return A->f32;
}

uint32_t g8(StructA *A, StructS *S, uint64_t count) {
  // CIR-LABEL: cir.func @_Z2g8
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructA_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u16i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u16i, !cir.ptr<!u16i> tbaa(#tbaa[[TAG_StructS_f16]])

  // LLVM-LABEL: define{{.*}} i32 @_Z2g8
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i16 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i16]]
  // PATH-LABEL: define{{.*}} i32 @_Z2g8
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_A_f32]]
  // PATH: store i16 4, ptr %{{.*}}, align {{4|2}}, !tbaa [[TAG_S_f16:!.*]]
  A->f32 = 1;
  S->f16 = 4;
  return A->f32;
}

uint32_t g9(StructS *S, StructS2 *S2, uint64_t count) {
  // CIR-LABEL: cir.func @_Z2g9
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructS_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructS2_f32]])

  // LLVM-LABEL: define{{.*}} i32 @_Z2g9
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH-LABEL: define{{.*}} i32 @_Z2g9
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_S_f32]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_S2_f32:!.*]]
  S->f32 = 1;
  S2->f32 = 4;
  return S->f32;
}

uint32_t g10(StructS *S, StructS2 *S2, uint64_t count) {
  // CIR-LABEL: cir.func @_Z3g10
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructS_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u16i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u16i, !cir.ptr<!u16i> tbaa(#tbaa[[TAG_StructS2_f16]])

  // LLVM-LABEL: define{{.*}} i32 @_Z3g10
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i16 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i16]]
  // PATH-LABEL: define{{.*}} i32 @_Z3g10
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_S_f32]]
  // PATH: store i16 4, ptr %{{.*}}, align {{4|2}}, !tbaa [[TAG_S2_f16:!.*]]
  S->f32 = 1;
  S2->f16 = 4;
  return S->f32;
}

uint32_t g11(StructC *C, StructD *D, uint64_t count) {
  // CIR-LABEL: cir.func @_Z3g11
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructC_b_a_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructD_b_a_f32]])

  // LLVM-LABEL: define{{.*}} i32 @_Z3g11
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH-LABEL: define{{.*}} i32 @_Z3g11
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_C_b_a_f32:!.*]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_D_b_a_f32:!.*]]
  C->b.a.f32 = 1;
  D->b.a.f32 = 4;
  return C->b.a.f32;
}

uint32_t g12(StructC *C, StructD *D, uint64_t count) {
  // CIR-LABEL: cir.func @_Z3g12
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructB_a_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructB_a_f32]])

  // LLVM-LABEL: define{{.*}} i32 @_Z3g12
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // TODO(cir): differentiate the two accesses.
  // PATH-LABEL: define{{.*}} i32 @_Z3g12
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_B_a_f32]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_B_a_f32]]
  StructB *b1 = &(C->b);
  StructB *b2 = &(D->b);
  // b1, b2 have different context.
  b1->a.f32 = 1;
  b2->a.f32 = 4;
  return b1->a.f32;
}

struct six {
  char a;
  int :0;
  char b;
  char c;
};
char g14(struct six *a, struct six *b) {
  // CIR-LABEL: cir.func @_Z3g14
  // CIR: %[[TMP1:.*]] = cir.load %{{.*}} : !cir.ptr<!cir.ptr<!ty_six>>, !cir.ptr<!ty_six>
  // CIR: %[[TMP2:.*]] = cir.get_member %[[TMP1]][2] {name = "b"} : !cir.ptr<!ty_six> -> !cir.ptr<!s8i>
  // CIR: %[[TMP3:.*]] = cir.load %[[TMP2]] : !cir.ptr<!s8i>, !s8i tbaa(#tbaa[[TAG_six_b]])

  // LLVM-LABEL: define{{.*}} i8 @_Z3g14
  // LLVM: load i8, ptr %{{.*}}, align 1, !tbaa [[TAG_char]]
  // PATH-LABEL: define{{.*}} i8 @_Z3g14
  // PATH: load i8, ptr %{{.*}}, align 1, !tbaa [[TAG_six_b:!.*]]
  return a->b;
}

// Types that differ only by name may alias.
typedef StructS StructS3;
uint32_t g15(StructS *S, StructS3 *S3, uint64_t count) {
  // CIR-LABEL: cir.func @_Z3g15
  // CIR: %[[INT_1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[UINT_1:.*]] = cir.cast(integral, %[[INT_1]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_1]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructS_f32]])
  // CIR: %[[INT_4:.*]] = cir.const #cir.int<4> : !s32i
  // CIR: %[[UINT_4:.*]] = cir.cast(integral, %[[INT_4]] : !s32i), !u32i
  // CIR: cir.store %[[UINT_4]], %{{.*}} : !u32i, !cir.ptr<!u32i> tbaa(#tbaa[[TAG_StructS_f32]])


  // LLVM-LABEL: define{{.*}} i32 @_Z3g15
  // LLVM: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // LLVM: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_i32]]
  // PATH-LABEL: define{{.*}} i32 @_Z3g15
  // PATH: store i32 1, ptr %{{.*}}, align 4, !tbaa [[TAG_S_f32]]
  // PATH: store i32 4, ptr %{{.*}}, align 4, !tbaa [[TAG_S_f32]]
  S->f32 = 1;
  S3->f32 = 4;
  return S->f32;
}

// LLVM: [[TYPE_char:!.*]] = !{!"omnipotent char", [[TAG_cxx_tbaa:!.*]],
// LLVM: [[TAG_cxx_tbaa]] = !{!"Simple C++ TBAA"}
// LLVM: [[TAG_i32]] = !{[[TYPE_i32:!.*]], [[TYPE_i32]], i64 0}
// LLVM: [[TYPE_i32]] = !{!"int", [[TYPE_char]],
// LLVM: [[TAG_i16]] = !{[[TYPE_i16:!.*]], [[TYPE_i16]], i64 0}
// LLVM: [[TYPE_i16]] = !{!"short", [[TYPE_char]],
// LLVM: [[TAG_char]] = !{[[TYPE_char]], [[TYPE_char]], i64 0}

// OLD-PATH: [[TAG_i32]] = !{[[TYPE_INT:!.*]], [[TYPE_INT]], i64 0}
// OLD-PATH: [[TYPE_INT]] = !{!"int", [[TYPE_CHAR:!.*]], i64 0}
// OLD-PATH: [[TYPE_CHAR]] = !{!"omnipotent char", [[TAG_cxx_tbaa:!.*]],
// OLD-PATH: [[TAG_cxx_tbaa]] = !{!"Simple C/C++ TBAA"}
// OLD-PATH: [[TAG_A_f32]] = !{[[TYPE_A:!.*]], [[TYPE_INT]], i64 4}
// OLD-PATH: [[TYPE_A]] = !{!"_ZTS7StructA", [[TYPE_SHORT:!.*]], i64 0, [[TYPE_INT]], i64 4, [[TYPE_SHORT]], i64 8, [[TYPE_INT]], i64 12}
// OLD-PATH: [[TYPE_SHORT:!.*]] = !{!"short", [[TYPE_CHAR]]
// OLD-PATH: [[TAG_A_f16]] = !{[[TYPE_A]], [[TYPE_SHORT]], i64 0}
// OLD-PATH: [[TAG_B_a_f32]] = !{[[TYPE_B:!.*]], [[TYPE_INT]], i64 8}
// OLD-PATH: [[TYPE_B]] = !{!"_ZTS7StructB", [[TYPE_SHORT]], i64 0, [[TYPE_A]], i64 4, [[TYPE_INT]], i64 20}
// OLD-PATH: [[TAG_B_a_f16]] = !{[[TYPE_B]], [[TYPE_SHORT]], i64 4}
// OLD-PATH: [[TAG_B_f32]] = !{[[TYPE_B]], [[TYPE_INT]], i64 20}
// OLD-PATH: [[TAG_B_a_f32_2]] = !{[[TYPE_B]], [[TYPE_INT]], i64 16}
// OLD-PATH: [[TAG_S_f32]] = !{[[TYPE_S:!.*]], [[TYPE_INT]], i64 4}
// OLD-PATH: [[TYPE_S]] = !{!"_ZTS7StructS", [[TYPE_SHORT]], i64 0, [[TYPE_INT]], i64 4}
// OLD-PATH: [[TAG_S_f16]] = !{[[TYPE_S]], [[TYPE_SHORT]], i64 0}
// OLD-PATH: [[TAG_S2_f32]] = !{[[TYPE_S2:!.*]], [[TYPE_INT]], i64 4}
// OLD-PATH: [[TYPE_S2]] = !{!"_ZTS8StructS2", [[TYPE_SHORT]], i64 0, [[TYPE_INT]], i64 4}
// OLD-PATH: [[TAG_S2_f16]] = !{[[TYPE_S2]], [[TYPE_SHORT]], i64 0}
// OLD-PATH: [[TAG_C_b_a_f32]] = !{[[TYPE_C:!.*]], [[TYPE_INT]], i64 12}
// OLD-PATH: [[TYPE_C]] = !{!"_ZTS7StructC", [[TYPE_SHORT]], i64 0, [[TYPE_B]], i64 4, [[TYPE_INT]], i64 28}
// OLD-PATH: [[TAG_D_b_a_f32]] = !{[[TYPE_D:!.*]], [[TYPE_INT]], i64 12}
// OLD-PATH: [[TYPE_D]] = !{!"_ZTS7StructD", [[TYPE_SHORT]], i64 0, [[TYPE_B]], i64 4, [[TYPE_INT]], i64 28, [[TYPE_CHAR]], i64 32}
// OLD-PATH: [[TAG_six_b]] = !{[[TYPE_six:!.*]], [[TYPE_CHAR]], i64 4}
// OLD-PATH: [[TYPE_six]] = !{!"_ZTS3six", [[TYPE_CHAR]], i64 0, [[TYPE_CHAR]], i64 4, [[TYPE_CHAR]], i64 5}
