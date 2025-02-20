// This is inspired from clang/test/CodeGen/tbaa.cpp, with both CIR and LLVM checks.
// g13 is not supported due to DiscreteBitFieldABI is NYI.
// see clang/lib/CIR/CodeGen/CIRRecordLayoutBuilder.cpp CIRRecordLowering::accumulateBitFields

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

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
  S->f32 = 1;
  S3->f32 = 4;
  return S->f32;
}
