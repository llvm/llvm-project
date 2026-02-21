// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +avx512vl -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -triple=x86_64-unknown-linux -target-feature +avx512vl -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -ffreestanding -triple=x86_64-unknown-linux -target-feature +avx512vl -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

__v4si  test_builtin_ia32_alignd128()
{
  // CIR-LABEL: _builtin_ia32_alignd128
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !s32i>

  // LLVM-LABEL: test_builtin_ia32_alignd128
  // LLVM: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>

  // OGCG-LABEL: test_builtin_ia32_alignd128
  // OGCG: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  __v4si  vec1 = {0, 1, 2, 3};
  __v4si  vec2 = {4, 5, 6, 7};
  return __builtin_ia32_alignd128(vec1, vec2, 0);
}

__v8si  test_builtin_ia32_alignd256()
{
  // CIR-LABEL: _builtin_ia32_alignd256
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !s32i>
  // LLVM-LABEL: test_builtin_ia32_alignd256
  // LLVM: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_builtin_ia32_alignd256
  // OGCG: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  __v8si  vec1 = {0, 1, 2, 3, 4, 5, 6, 7};
  __v8si  vec2 = {8, 9, 10, 11, 12, 13, 14, 15};
  return __builtin_ia32_alignd256(vec1, vec2, 0);
}

__v16si  test_builtin_ia32_alignd512()
{
  // CIR-LABEL: _builtin_ia32_alignd512
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<16 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<16 x !s32i>
  // LLVM-LABEL: test_builtin_ia32_alignd512
  // LLVM: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // OGCG-LABEL: test_builtin_ia32_alignd512
  // OGCG: shufflevector <16 x i32> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  __v16si  vec1 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  __v16si  vec2 = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  return __builtin_ia32_alignd512(vec1, vec2, 0);
}

__v2di test_builtin_ia32_alignq128()
{
  // CIR-LABEL: _builtin_ia32_alignq128
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<2 x !s64i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i] : !cir.vector<2 x !s64i>
  // LLVM-LABEL: test_builtin_ia32_alignq128
  // LLVM: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 1>
  // OGCG-LABEL: test_builtin_ia32_alignq128
  // OGCG: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> <i32 0, i32 1>
  __v2di vec1 = {0, 1};
  __v2di vec2 = {2, 3};
  return __builtin_ia32_alignq128(vec1, vec2, 0);
}

__v4di test_builtin_ia32_alignq256()
{
  // CIR-LABEL: _builtin_ia32_alignq256
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<4 x !s64i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !s64i>
  // LLVM-LABEL: test_builtin_ia32_alignq256
  // LLVM: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG-LABEL: test_builtin_ia32_alignq256
  // OGCG: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  __v4di vec1 = {0, 1, 2, 3};
  __v4di vec2 = {4, 5, 6, 7};
  return __builtin_ia32_alignq256(vec1, vec2, 0);
}

__v8di test_builtin_ia32_alignq512()
{
  // CIR-LABEL: _builtin_ia32_alignq512
  // CIR: cir.vec.shuffle({{%.*}}, {{%.*}} : !cir.vector<8 x !s64i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !s64i>
  // LLVM-LABEL: test_builtin_ia32_alignq512
  // LLVM: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-LABEL: test_builtin_ia32_alignq512
  // OGCG: shufflevector <8 x i64> %{{.*}}, <8 x i64> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  __v8di vec1 = {0, 1, 2, 3, 4, 5, 6, 7};
  __v8di vec2 = {8, 9, 10, 11, 12, 13, 14, 15};
  return __builtin_ia32_alignq512(vec1, vec2, 0);
}
