// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -disable-O0-optnone -fclangir -emit-cir -o %t.cir | opt -S -passes=mem2reg
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-unknown-linux -target-feature +avx -disable-O0-optnone -fclangir -emit-cir -o %t.cir | opt -S -passes=mem2reg
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -disable-O0-optnone -fclangir -emit-llvm -o %t.ll | opt -S -passes=mem2reg
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-unknown-linux -target-feature +avx -disable-O0-optnone -fclangir -emit-llvm -o %t.ll | opt -S -passes=mem2reg
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -disable-O0-optnone -emit-llvm -o - | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-unknown-linux -target-feature +avx -disable-O0-optnone -emit-llvm -o - | opt -S -passes=mem2reg | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m256d test0_mm256_insertf128_pd(__m256d a, __m128d b) {
  // CIR-LABEL: @test0_mm256_insertf128_pd(
  // CIR: [[A:%.*]] = cir.load align(32) %0 : !cir.ptr<!cir.vector<4 x !cir.double>>, !cir.vector<4 x !cir.double>
  // CIR: [[B:%.*]] = cir.load align(16) %1 : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double>
  // CIR: %{{.*}} = cir.vec.shuffle([[B]], %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.double>
  // CIR-NEXT: %{{.*}} = cir.vec.shuffle([[A]], %{{.*}} : !s32i, #cir.int<5> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.double>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: @test0_mm256_insertf128_pd
  // LLVM:    [[A:%.*]] = load <4 x double>, ptr %{{.*}}, align 32
  // LLVM:    [[B:%.*]] = load <2 x double>, ptr %{{.*}}, align 16
  // LLVM-NEXT:    [[WIDEN:%.*]] = shufflevector <2 x double> [[B]], <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM-NEXT:    [[INSERT:%.*]] = shufflevector <4 x double> [[A]], <4 x double> [[WIDEN]], <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  // LLVM:    ret <4 x double>

  // OGCG-LABEL: define dso_local <4 x double> @test0_mm256_insertf128_pd(
  // OGCG-SAME: <4 x double> noundef [[A:%.*]], <2 x double> noundef [[B:%.*]]) #[[ATTR0:[0-9]+]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[WIDEN:%.*]] = shufflevector <2 x double> [[B]], <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG-NEXT:    [[INSERT:%.*]] = shufflevector <4 x double> [[A]], <4 x double> [[WIDEN]], <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  // OGCG-NEXT:    ret <4 x double> [[INSERT]]
  return _mm256_insertf128_pd(a, b, 0);
}

__m256d test1_mm256_insertf128_pd(__m256d a, __m128d b) {
  // CIR-LABEL: @test1_mm256_insertf128_pd(
  // CIR: [[A:%.*]] = cir.load align(32) %0 : !cir.ptr<!cir.vector<4 x !cir.double>>, !cir.vector<4 x !cir.double>
  // CIR: [[B:%.*]] = cir.load align(16) %1 : !cir.ptr<!cir.vector<2 x !cir.double>>, !cir.vector<2 x !cir.double>
  // CIR: %{{.*}} = cir.vec.shuffle([[B]], %{{.*}} : !cir.vector<2 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i] : !cir.vector<4 x !cir.double>
  // CIR-NEXT: %{{.*}} = cir.vec.shuffle([[A]], %{{.*}} : !cir.vector<4 x !cir.double>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i] : !cir.vector<4 x !cir.double>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: @test1_mm256_insertf128_pd
  // LLVM:    [[A:%.*]] = load <4 x double>, ptr %{{.*}}, align 32
  // LLVM:    [[B:%.*]] = load <2 x double>, ptr %{{.*}}, align 16
  // LLVM-NEXT:    [[WIDEN:%.*]] = shufflevector <2 x double> [[B]], <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // LLVM-NEXT:    [[INSERT:%.*]] = shufflevector <4 x double> [[A]], <4 x double> [[WIDEN]], <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  // LLVM:    ret <4 x double>

  // OGCG-LABEL: define dso_local <4 x double> @test1_mm256_insertf128_pd(
  // OGCG-SAME: <4 x double> noundef [[A:%.*]], <2 x double> noundef [[B:%.*]]) #[[ATTR0]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[WIDEN:%.*]] = shufflevector <2 x double> [[B]], <2 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  // OGCG-NEXT:    [[INSERT:%.*]] = shufflevector <4 x double> [[A]], <4 x double> [[WIDEN]], <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  // OGCG-NEXT:    ret <4 x double> [[INSERT]]
  return _mm256_insertf128_pd(a, b, 1);
}

__m256 test0_mm256_insertf128_ps(__m256 a, __m128 b) {
  // CIR-LABEL: @test0_mm256_insertf128_ps(
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.float>
  // CIR-NEXT: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.float>) [#cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.float>

  // LLVM-LABEL: @test0_mm256_insertf128_ps(
  // LLVM:    %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    %{{.*}} = shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
  // LLVM:    ret <8 x float> %{{.*}}

  // OGCG-LABEL: define dso_local <8 x float> @test0_mm256_insertf128_ps(
  // OGCG-SAME: <8 x float> noundef [[A:%.*]], <4 x float> noundef [[B:%.*]]) #[[ATTR0]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[WIDEN:%.*]] = shufflevector <4 x float> [[B]], <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-NEXT:    [[INSERT:%.*]] = shufflevector <8 x float> [[A]], <8 x float> [[WIDEN]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
  // OGCG-NEXT:    ret <8 x float> [[INSERT]]
  return _mm256_insertf128_ps(a, b, 0);
}

__m256 test1_mm256_insertf128_ps(__m256 a, __m128 b) {
  // CIR-LABEL: @test1_mm256_insertf128_ps(
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !cir.float>
  // CIR-NEXT: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<8 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.float>

  // LLVM-LABEL: define dso_local <8 x float> @test1_mm256_insertf128_ps(
  // LLVM:    %{{.*}} = shufflevector <4 x float> %{{.*}}, <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    %{{.*}} = shufflevector <8 x float> %{{.*}}, <8 x float> %{{.*}}, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // LLVM:    ret <8 x float> %{{.*}}

  // OGCG-LABEL: define dso_local <8 x float> @test1_mm256_insertf128_ps(
  // OGCG-SAME: <8 x float> noundef [[A:%.*]], <4 x float> noundef [[B:%.*]]) #[[ATTR0]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[WIDEN:%.*]] = shufflevector <4 x float> [[B]], <4 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-NEXT:    [[INSERT:%.*]] = shufflevector <8 x float> [[A]], <8 x float> [[WIDEN]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // OGCG-NEXT:    ret <8 x float> [[INSERT]]
  return _mm256_insertf128_ps(a, b, 1);
}

__m256i test0_mm256_insertf128_si256(__m256i a, __m128i b) {
  // CIR-LABEL: @test0_mm256_insertf128_si256(
  // CIR: [[TMP0:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<4 x !s64i> -> !cir.vector<8 x !s32i>
  // CIR: [[TMP1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<4 x !s32i>
  // CIR: %{{.*}} = cir.vec.shuffle([[TMP1]], %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !s32i>
  // CIR-NEXT: %{{.*}} = cir.vec.shuffle([[TMP0]], %{{.*}} : !cir.vector<8 x !s32i>) [#cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !s32i>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s32i> -> !cir.vector<4 x !s64i>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !s64i>

  // LLVM-LABEL: @test0_mm256_insertf128_si256
  // LLVM:    [[TMP0:%.*]] = bitcast <4 x i64> %{{.*}} to <8 x i32>
  // LLVM:    [[TMP1:%.*]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // LLVM:    [[WIDEN:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[INSERT:%.*]] = shufflevector <8 x i32> [[TMP0]], <8 x i32> [[WIDEN]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
  // LLVM:    [[TMP2:%.*]] = bitcast <8 x i32> [[INSERT]] to <4 x i64>
  // LLVM:    ret <4 x i64> %{{.*}}

  // OGCG-LABEL: define dso_local <4 x i64> @test0_mm256_insertf128_si256(
  // OGCG-SAME: <4 x i64> noundef [[A:%.*]], <2 x i64> noundef [[B:%.*]]) #[[ATTR0]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[TMP0:%.*]] = bitcast <4 x i64> [[A]] to <8 x i32>
  // OGCG-NEXT:    [[TMP1:%.*]] = bitcast <2 x i64> [[B]] to <4 x i32>
  // OGCG-NEXT:    [[WIDEN:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-NEXT:    [[INSERT:%.*]] = shufflevector <8 x i32> [[TMP0]], <8 x i32> [[WIDEN]], <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7>
  // OGCG-NEXT:    [[TMP2:%.*]] = bitcast <8 x i32> [[INSERT]] to <4 x i64>
  // OGCG-NEXT:    ret <4 x i64> [[TMP2]]
  return _mm256_insertf128_si256(a, b, 0);
}

__m256i test1_mm256_insertf128_si256(__m256i a, __m128i b) {
  // CIR-LABEL: @test1_mm256_insertf128_si256(
  // CIR: [[TMP0:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<4 x !s64i> -> !cir.vector<8 x !s32i>
  // CIR: [[TMP1:%.*]] = cir.cast bitcast %{{.*}} : !cir.vector<2 x !s64i> -> !cir.vector<4 x !s32i>
  // CIR: %{{.*}} = cir.vec.shuffle([[TMP1]], %{{.*}} : !cir.vector<4 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i] : !cir.vector<8 x !s32i>
  // CIR-NEXT: %{{.*}} = cir.vec.shuffle([[TMP0]], %{{.*}} : !cir.vector<8 x !s32i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i] : !cir.vector<8 x !s32i>
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !cir.vector<8 x !s32i> -> !cir.vector<4 x !s64i>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !s64i>

  // LLVM-LABEL: @test1_mm256_insertf128_si256
  // LLVM:    [[TMP0:%.*]] = bitcast <4 x i64> %{{.*}} to <8 x i32>
  // LLVM:    [[TMP1:%.*]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // LLVM:    [[WIDEN:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // LLVM-NEXT:    [[INSERT:%.*]] = shufflevector <8 x i32> [[TMP0]], <8 x i32> [[WIDEN]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // LLVM:    [[TMP2:%.*]] = bitcast <8 x i32> [[INSERT]] to <4 x i64>
  // LLVM:    ret <4 x i64> %{{.*}}

  // OGCG-LABEL: define dso_local <4 x i64> @test1_mm256_insertf128_si256(
  // OGCG-SAME: <4 x i64> noundef [[A:%.*]], <2 x i64> noundef [[B:%.*]]) #[[ATTR0]] {
  // OGCG-NEXT:  [[ENTRY:.*:]]
  // OGCG-NEXT:    [[TMP0:%.*]] = bitcast <4 x i64> [[A]] to <8 x i32>
  // OGCG-NEXT:    [[TMP1:%.*]] = bitcast <2 x i64> [[B]] to <4 x i32>
  // OGCG-NEXT:    [[WIDEN:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // OGCG-NEXT:    [[INSERT:%.*]] = shufflevector <8 x i32> [[TMP0]], <8 x i32> [[WIDEN]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  // OGCG-NEXT:    [[TMP2:%.*]] = bitcast <8 x i32> [[INSERT]] to <4 x i64>
  // OGCG-NEXT:    ret <4 x i64> [[TMP2]]
  return _mm256_insertf128_si256(a, b, 1);
}
