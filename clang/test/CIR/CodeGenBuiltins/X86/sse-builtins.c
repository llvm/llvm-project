// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

// This test mimics clang/test/CodeGen/X86/sse-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

void test_mm_setcsr(unsigned int A) {
  // CIR-LABEL: test_mm_setcsr
  // CIR: cir.store {{.*}}, {{.*}} : !u32i
  // CIR: cir.call_llvm_intrinsic "x86.sse.ldmxcsr" {{.*}} : (!cir.ptr<!u32i>) -> !void

  // LLVM-LABEL: test_mm_setcsr
  // LLVM: store i32
  // LLVM: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})

  // OGCG-LABEL: test_mm_setcsr
  // OGCG: store i32
  // OGCG: call void @llvm.x86.sse.ldmxcsr(ptr {{.*}})
  _mm_setcsr(A);
}

unsigned int test_mm_getcsr(void) {
  // CIR-LABEL: test_mm_getcsr
  // CIR: cir.call_llvm_intrinsic "x86.sse.stmxcsr" %{{.*}} : (!cir.ptr<!u32i>) -> !void
  // CIR: cir.load {{.*}} : !cir.ptr<!u32i>, !u32i

  // LLVM-LABEL: test_mm_getcsr
  // LLVM: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // LLVM: load i32

  // OGCG-LABEL: test_mm_getcsr
  // OGCG: call void @llvm.x86.sse.stmxcsr(ptr %{{.*}})
  // OGCG: load i32
  return _mm_getcsr();
}

void test_mm_sfence(void) {
  // CIR-LABEL: test_mm_sfence
  // LLVM-LABEL: test_mm_sfence
  // OGCG-LABEL: test_mm_sfence
  _mm_sfence();
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.sse.sfence" : () -> !void
  // LLVM: call void @llvm.x86.sse.sfence()
  // OGCG: call void @llvm.x86.sse.sfence()
}

void test_mm_prefetch(char const* p) {
  // CIR-LABEL: test_mm_prefetch
  // LLVM-LABEL: test_mm_prefetch
  // OGCG-LABEL: test_mm_prefetch
  _mm_prefetch(p, 0);
  // CIR: cir.prefetch read locality(0) %{{.*}} : !cir.ptr<!void>
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 0, i32 1)
  // OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 0, i32 1)
}

void test_mm_prefetch_local(char const* p) {
  // CIR-LABEL: test_mm_prefetch_local
  // LLVM-LABEL: test_mm_prefetch_local
  // OGCG-LABEL: test_mm_prefetch_local
  _mm_prefetch(p, 3);
  // CIR: cir.prefetch read locality(3) %{{.*}} : !cir.ptr<!void>
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
  // OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
}

void test_mm_prefetch_write(char const* p) {
  // CIR-LABEL: test_mm_prefetch_write
  // LLVM-LABEL: test_mm_prefetch_write
  // OGCG-LABEL: test_mm_prefetch_write
  _mm_prefetch(p, 7);
  // CIR: cir.prefetch write locality(3) %{{.*}} : !cir.ptr<!void>
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 3, i32 1)
  // OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 3, i32 1)
}

__m128 test_mm_undefined_ps(void) {
  // CIR-LABEL: _mm_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<2 x !cir.double> -> !cir.vector<4 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_undefined_ps
  // LLVM: store <4 x float> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <4 x float>, ptr %[[A]], align 16
  // LLVM: ret <4 x float> %{{.*}}

  // OGCG-LABEL: test_mm_undefined_ps
  // OGCG: ret <4 x float> zeroinitializer
  return _mm_undefined_ps();
}

__m128 test_mm_shuffle_ps(__m128 A, __m128 B) {
  // CIR-LABEL: _mm_shuffle_ps
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<4 x !cir.float>) [#cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i] : !cir.vector<4 x !cir.float>

  // LLVM-LABEL: test_mm_shuffle_ps
  // LLVM: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 0, i32 4, i32 4>

  // OGCG-LABEL: test_mm_shuffle_ps
  // OGCG: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x i32> <i32 0, i32 0, i32 4, i32 4>
  return _mm_shuffle_ps(A, B, 0);
}

__m128i test_mm_slli_si128(__m128i a) {
  // CIR-LABEL: test_mm_slli_si128
  // CIR: [[B:%.*]] = cir.cast bitcast {{.*}} : !cir.vector<2 x !{{.*}}64i> -> !cir.vector<16 x !{{.*}}8i>
  // CIR: [[ZERO:%.*]] = cir.const #cir.zero : !cir.vector<16 x !{{.*}}8i>
  // CIR: [[PSLLDQ:%.*]] = cir.vec.shuffle([[ZERO]], [[B]] : !cir.vector<16 x !{{.*}}8i>) [#cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i] : !cir.vector<16 x !{{.*}}8i>
  // CIR: {{.*}} = cir.cast bitcast [[PSLLDQ]] : !cir.vector<16 x !{{.*}}8i> -> !cir.vector<2 x !{{.*}}64i>
  // CIR: cir.return {{.*}} : !cir.vector<2 x !{{.*}}64i>
  
  // LLVM-LABEL: test_mm_slli_si128
  // LLVM: [[CAST:%.*]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // LLVM: [[PSLLDQ:%.*]] = shufflevector <16 x i8> zeroinitializer, <16 x i8> [[CAST]], <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  // LLVM: {{.*}} = bitcast <16 x i8> [[PSLLDQ]] to <2 x i64>
  // LLVM: ret <2 x i64> {{.*}}
  
  // OGCG-LABEL: test_mm_slli_si128
  // OGCG: [[CAST:%.*]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // OGCG: [[PSLLDQ:%.*]] = shufflevector <16 x i8> zeroinitializer, <16 x i8> [[CAST]], <16 x i32> <i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26>
  // OGCG: [[CAST1:%.*]] = bitcast <16 x i8> [[PSLLDQ]] to <2 x i64>
  // OGCG: ret <2 x i64> [[CAST1]]
  return _mm_slli_si128(a, 5);
}

__m128i test_mm_slli_si128_0(__m128i a) {
  // CIR-LABEL: test_mm_slli_si128_0 
  // CIR: {{.*}} = cir.vec.shuffle({{.*}}, {{.*}} : !cir.vector<16 x !{{.*}}8i>) [#cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i] : !cir.vector<16 x !{{.*}}8i>
  // CIR: {{.*}} = cir.cast bitcast {{.*}} : !cir.vector<16 x !{{.*}}8i> -> !cir.vector<2 x !{{.*}}64i>
  // CIR: cir.return {{.*}} : !cir.vector<2 x !{{.*}}64i>
  
  // LLVM-LABEL test_mm_slli_si128_0
  // LLVM: [[CAST]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // LLVM: [[PSLLDQ:%.*]] = shufflevector <16 x i8> zeroinitializer, <16 x i8> [[CAST]], <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // LLVM: {{.*}} = bitcast <16 x i8> [[PSLLDQ]] to <2 x i64>
  // LLVM: ret <2 x i64> {{.*}}

  // OGCG-LABEL test_mm_slli_si128_0
  // OGCG: [[CAST]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // OGCG: [[PSLLDQ:%.*]] = shufflevector <16 x i8> zeroinitializer, <16 x i8> [[CAST]], <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // OGCG: [[CAST1:%.*]] = bitcast <16 x i8> [[PSLLDQ]] to <2 x i64>
  // OGCG: ret <2 x i64> [[CAST1]]
  return _mm_slli_si128(a, 0);
}

__m128i test_mm_slli_si128_16(__m128i a) {
  // CIR-LABEL: test_mm_slli_si128_16
  // CIR: [[ZERO:%.*]] = cir.const #cir.zero : !cir.vector<16 x !{{.*}}8i> 
  // CIR: {{.*}} = cir.cast bitcast [[ZERO]] : !cir.vector<16 x !{{.*}}8i> -> !cir.vector<2 x !s64i>
  // CIR: cir.return {{.*}} : !cir.vector<2 x !{{.*}}64i>
  
  // LLVM-LABEL: test_mm_slli_si128_16
  // LLVM: store <2 x i64> zeroinitializer, ptr [[ZERO:%.*]]
  // LLVM: [[RET:%.*]] = load <2 x i64>, ptr [[ZERO]]
  // LLVM: ret <2 x i64> [[RET]] 
  
  // OGCG-LABEL: test_mm_slli_si128_16
  // OGCG: [[CAST:%.*]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // OGCG: ret <2 x i64> zeroinitializer
  return _mm_slli_si128(a, 16);
}

__m128i test_mm_srli_si128(__m128i a) {
  // CIR-LABEL: test_mm_srli_si128
  // CIR: [[A:%.*]] = cir.cast bitcast {{.*}} : !cir.vector<2 x !{{.*}}64i> -> !cir.vector<16 x !{{.*}}8i>
  // CIR: [[ZERO:%.*]] = cir.const #cir.zero : !cir.vector<16 x !{{.*}}8i>
  // CIR: [[PSRLDQ:%.*]] = cir.vec.shuffle([[A]], [[ZERO]] : !cir.vector<16 x !{{.*}}8i>) [#cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i] : !cir.vector<16 x !{{.*}}8i>
  // CIR: {{.*}} = cir.cast bitcast [[PSRLDQ]] : !cir.vector<16 x !{{.*}}8i> -> !cir.vector<2 x !{{.*}}64i>
  // CIR: cir.return {{.*}} : !cir.vector<2 x !{{.*}}64i>
  
  // LLVM-LABEL: test_mm_srli_si128
  // LLVM: [[CAST:%.*]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // LLVM: [[PSRLDQ:%.*]] = shufflevector <16 x i8> [[CAST]], <16 x i8> zeroinitializer, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  // LLVM: {{.*}} = bitcast <16 x i8> [[PSRLDQ]] to <2 x i64>
  // LLVM: ret <2 x i64> {{.*}} 

  // OGCG-LABEL: test_mm_srli_si128
  // OGCG: [[CAST:%.*]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // OGCG: [[PSRLDQ:%.*]] = shufflevector <16 x i8> [[CAST]], <16 x i8> zeroinitializer, <16 x i32> <i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20>
  // OGCG: [[CAST1:%.*]] = bitcast <16 x i8> [[PSRLDQ]] to <2 x i64>
  // OGCG: ret <2 x i64> [[CAST1]]
  return _mm_srli_si128(a, 5);
}

__m128i test_mm_srli_si128_0(__m128i a) {
  // CIR-LABEL: test_mm_srli_si128_0
  // CIR: [[A:%.*]] = cir.cast bitcast {{.*}} : !cir.vector<2 x !{{.*}}64i> -> !cir.vector<16 x !{{.*}}8i>
  // CIR: [[ZERO:%.*]] = cir.const #cir.zero : !cir.vector<16 x !{{.*}}8i>
  // CIR: [[PSRLDQ:%.*]] = cir.vec.shuffle([[A]], [[ZERO]] : !cir.vector<16 x !{{.*}}8i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i] : !cir.vector<16 x !{{.*}}8i>
  // CIR: {{.*}} = cir.cast bitcast [[PSRLDQ]] : !cir.vector<16 x !{{.*}}8i> -> !cir.vector<2 x !{{.*}}64i>
  // CIR: cir.return {{.*}} : !cir.vector<2 x !{{.*}}64i>
  
  // LLVM-LABEL: test_mm_srli_si128_0
  // LLVM: [[CAST:%.*]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // LLVM: [[PSRLDQ:%.*]] = shufflevector <16 x i8> [[CAST]], <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: {{.*}} = bitcast <16 x i8> [[PSRLDQ]] to <2 x i64>
  // LLVM: ret <2 x i64> {{.*}}

  // OGCG-LABEL: test_mm_srli_si128_0
  // OGCG: [[CAST:%.*]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // OGCG: [[PSRLDQ:%.*]] = shufflevector <16 x i8> [[CAST]], <16 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // OGCG: [[CAST1:%.*]] = bitcast <16 x i8> [[PSRLDQ]] to <2 x i64>
  // OGCG: ret <2 x i64> [[CAST1]]
  return _mm_srli_si128(a, 0);
}

__m128i test_mm_srli_si128_16(__m128i a) {
  // CIR-LABEL: test_mm_srli_si128_16
  // CIR: [[ZERO:%.*]] = cir.const #cir.zero : !cir.vector<16 x !{{.*}}8i> 
  // CIR: {{.*}} = cir.cast bitcast [[ZERO]] : !cir.vector<16 x !{{.*}}8i> -> !cir.vector<2 x !s64i>
  // CIR: cir.return {{.*}} : !cir.vector<2 x !{{.*}}64i>
  
  // LLVM-LABEL: test_mm_srli_si128_16
  // LLVM: store <2 x i64> zeroinitializer, ptr [[ZERO:%.*]]
  // LLVM: [[RET:%.*]] = load <2 x i64>, ptr [[ZERO]]
  // LLVM: ret <2 x i64> [[RET]] 
  
  // OGCG-LABEL: test_mm_srli_si128_16
  // OGCG: [[CAST:%.*]] = bitcast <2 x i64> {{.*}} to <16 x i8>
  // OGCG: ret <2 x i64> zeroinitializer
  return _mm_srli_si128(a, 16);
}
