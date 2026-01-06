// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -target-feature +avx  -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse4.1 -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

typedef short __v4hi __attribute__((__vector_size__(8)));
typedef char __v16qi __attribute__((__vector_size__(16)));
typedef short __v8hi __attribute__((__vector_size__(16)));
typedef int __v4si __attribute__((__vector_size__(16)));
typedef long long __v2di __attribute__((__vector_size__(16)));
typedef char __v32qi __attribute__((__vector_size__(32)));
typedef short __v16hi __attribute__((__vector_size__(32)));
typedef int __v8si __attribute__((__vector_size__(32)));
typedef long long __v4di __attribute__((__vector_size__(32)));

__v4hi test_vec_set_v4hi(__v4hi a, short b) {
  // CIR-LABEL: test_vec_set_v4hi
  // CIR: {{%.*}} = cir.const #cir.int<2> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !s16i>

  // LLVM-LABEL: test_vec_set_v4hi
  // LLVM: {{%.*}} = insertelement <4 x i16> {{%.*}}, i16 {{%.*}}, i64 2

  // OGCG-LABEL: test_vec_set_v4hi
  // OGCG: {{%.*}} = insertelement <4 x i16> {{%.*}}, i16 {{%.*}}, i64 2
  return __builtin_ia32_vec_set_v4hi(a, b, 2);
}

__v16qi test_vec_set_v16qi(__v16qi a, char b) {
  // CIR-LABEL: test_vec_set_v16qi
  // CIR: {{%.*}} = cir.const #cir.int<5> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<16 x !s8i>

  // LLVM-LABEL: test_vec_set_v16qi
  // LLVM: {{%.*}} = insertelement <16 x i8> {{%.*}}, i8 {{%.*}}, i64 5

  // OGCG-LABEL: test_vec_set_v16qi
  // OGCG: {{%.*}} = insertelement <16 x i8> {{%.*}}, i8 {{%.*}}, i64 5
  return __builtin_ia32_vec_set_v16qi(a, b, 5);
}

__v8hi test_vec_set_v8hi(__v8hi a, short b) {
  // CIR-LABEL: test_vec_set_v8hi
  // CIR: {{%.*}} = cir.const #cir.int<3> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<8 x !s16i>

  // LLVM-LABEL: test_vec_set_v8hi
  // LLVM: {{%.*}} = insertelement <8 x i16> {{%.*}}, i16 {{%.*}}, i64 3

  // OGCG-LABEL: test_vec_set_v8hi
  // OGCG: {{%.*}} = insertelement <8 x i16> {{%.*}}, i16 {{%.*}}, i64 3
  return __builtin_ia32_vec_set_v8hi(a, b, 3);
}

__v4si test_vec_set_v4si(__v4si a, int b) {
  // CIR-LABEL: test_vec_set_v4si
  // CIR: {{%.*}} = cir.const #cir.int<1> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !s32i>

  // LLVM-LABEL: test_vec_set_v4si
  // LLVM: {{%.*}} = insertelement <4 x i32> {{%.*}}, i32 {{%.*}}, i64 1

  // OGCG-LABEL: test_vec_set_v4si
  // OGCG: {{%.*}} = insertelement <4 x i32> {{%.*}}, i32 {{%.*}}, i64 1
  return __builtin_ia32_vec_set_v4si(a, b, 1);
}

__v2di test_vec_set_v2di(__v2di a, long long b) {
  // CIR-LABEL: test_vec_set_v2di
  // CIR: {{%.*}} = cir.const #cir.int<0> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<2 x !s64i>

  // LLVM-LABEL: test_vec_set_v2di
  // LLVM: {{%.*}} = insertelement <2 x i64> {{%.*}}, i64 {{%.*}}, i64 0

  // OGCG-LABEL: test_vec_set_v2di
  // OGCG: {{%.*}} = insertelement <2 x i64> {{%.*}}, i64 {{%.*}}, i64 0
  return __builtin_ia32_vec_set_v2di(a, b, 0);
}

__v32qi test_vec_set_v32qi(__v32qi a, char b) {
  // CIR-LABEL: test_vec_set_v32qi
  // CIR: {{%.*}} = cir.const #cir.int<10> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<32 x !s8i>

  // LLVM-LABEL: test_vec_set_v32qi
  // LLVM: {{%.*}} = insertelement <32 x i8> {{%.*}}, i8 {{%.*}}, i64 10

  // OGCG-LABEL: test_vec_set_v32qi
  // OGCG: {{%.*}} = insertelement <32 x i8> {{%.*}}, i8 {{%.*}}, i64 10
  return __builtin_ia32_vec_set_v32qi(a, b, 10);
}

__v16hi test_vec_set_v16hi(__v16hi a, short b) {
  // CIR-LABEL: test_vec_set_v16hi
  // CIR: {{%.*}} = cir.const #cir.int<7> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<16 x !s16i>

  // LLVM-LABEL: test_vec_set_v16hi
  // LLVM: {{%.*}} = insertelement <16 x i16> {{%.*}}, i16 {{%.*}}, i64 7

  // OGCG-LABEL: test_vec_set_v16hi
  // OGCG: {{%.*}} = insertelement <16 x i16> {{%.*}}, i16 {{%.*}}, i64 7
  return __builtin_ia32_vec_set_v16hi(a, b, 7);
}

__v8si test_vec_set_v8si(__v8si a, int b) {
  // CIR-LABEL: test_vec_set_v8si
  // CIR: {{%.*}} = cir.const #cir.int<4> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<8 x !s32i>

  // LLVM-LABEL: test_vec_set_v8si
  // LLVM: {{%.*}} = insertelement <8 x i32> {{%.*}}, i32 {{%.*}}, i64 4

  // OGCG-LABEL: test_vec_set_v8si
  // OGCG: {{%.*}} = insertelement <8 x i32> {{%.*}}, i32 {{%.*}}, i64 4
  return __builtin_ia32_vec_set_v8si(a, b, 4);
}

__v4di test_vec_set_v4di(__v4di a, long long b) {
  // CIR-LABEL: test_vec_set_v4di
  // CIR: {{%.*}} = cir.const #cir.int<2> : !u64i
  // CIR: {{%.*}} = cir.vec.insert %{{.*}}, %{{.*}}[%{{.*}} : !u64i] : !cir.vector<4 x !s64i>

  // LLVM-LABEL: test_vec_set_v4di
  // LLVM: {{%.*}} = insertelement <4 x i64> {{%.*}}, i64 {{%.*}}, i64 2

  // OGCG-LABEL: test_vec_set_v4di
  // OGCG: {{%.*}} = insertelement <4 x i64> {{%.*}}, i64 {{%.*}}, i64 2
  return __builtin_ia32_vec_set_v4di(a, b, 2);
}
