// RUN: %clang_cc1 %s -triple=x86_64-unknown-linux -target-feature +ssse3 -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=CIR

// RUN: %clang_cc1 %s -triple=x86_64-unknown-linux -target-feature +ssse3 -emit-llvm -o %t_og.ll
// RUN: FileCheck --input-file=%t_og.ll %s --check-prefix=OGCG

#define _mm_alignr_epi8(a, b, n) (__builtin_ia32_palignr128((a), (b), (n)))
typedef __attribute__((vector_size(16))) int int4;

// CIR-LABEL: @align1
// CIR: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 15, i32 16, i32 17
// OGCG-LABEL: @align1
// OGCG: %palignr = shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 15, i32 16, i32 17
int4 align1(int4 a, int4 b) { return _mm_alignr_epi8(a, b, 15); }

// CIR-LABEL: @align2
// CIR: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 16, i32 17, i32 18
// OGCG-LABEL: @align2
// OGCG: %palignr = shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> <i32 16, i32 17, i32 18
int4 align2(int4 a, int4 b) { return _mm_alignr_epi8(a, b, 16); }

// CIR-LABEL: @align3
// CIR: shufflevector <16 x i8> %{{.*}}, <16 x i8> zeroinitializer, <16 x i32> <i32 1, i32 2, i32 3
// OGCG-LABEL: @align3  
// OGCG: %palignr = shufflevector <16 x i8> %{{.*}}, <16 x i8> zeroinitializer, <16 x i32> <i32 1, i32 2, i32 3
int4 align3(int4 a, int4 b) { return _mm_alignr_epi8(a, b, 17); }

// CIR-LABEL: @align4
// CIR: store <4 x i32> zeroinitializer, ptr %{{.*}}, align 16
// OGCG-LABEL: @align4
// OGCG: ret <4 x i32> zeroinitializer
int4 align4(int4 a, int4 b) { return _mm_alignr_epi8(a, b, 32); }
