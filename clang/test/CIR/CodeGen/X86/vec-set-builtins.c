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

#include <immintrin.h>

typedef short __v4hi __attribute__((__vector_size__(8)));
typedef char __v16qi __attribute__((__vector_size__(16)));
typedef short __v8hi __attribute__((__vector_size__(16)));
typedef int __v4si __attribute__((__vector_size__(16)));
typedef long long __v2di __attribute__((__vector_size__(16)));

__v4hi test_vec_set_v4hi(__v4hi a, short b) {
  // CIR-LABEL: cir.func{{.*}}@test_vec_set_v4hi
  // CIR: %[[VEC:.*]] = cir.load
  // CIR: %[[VAL:.*]] = cir.load
  // CIR: %[[IDX:.*]] = cir.const #cir.int<2> : !s64i
  // CIR: %[[RESULT:.*]] = cir.vec.insert %[[VAL]], %[[VEC]][%[[IDX]] : !s64i] : !cir.vector<4 x !s16i>
  // CIR: cir.return %[[RESULT]]

  // LLVM-LABEL: @test_vec_set_v4hi
  // LLVM: %[[VEC:.*]] = load <4 x i16>
  // LLVM: %[[VAL:.*]] = load i16
  // LLVM: %[[RESULT:.*]] = insertelement <4 x i16> %[[VEC]], i16 %[[VAL]], i64 2
  // LLVM: ret <4 x i16> %[[RESULT]]

  // OGCG-LABEL: @test_vec_set_v4hi
  // OGCG: %[[VEC:.*]] = load <4 x i16>
  // OGCG: %[[VAL:.*]] = load i16
  // OGCG: %[[RESULT:.*]] = insertelement <4 x i16> %[[VEC]], i16 %[[VAL]], i64 2
  // OGCG: ret <4 x i16> %[[RESULT]]
  return __builtin_ia32_vec_set_v4hi(a, b, 2);
}

__v16qi test_vec_set_v16qi(__v16qi a, char b) {
  // CIR-LABEL: cir.func{{.*}}@test_vec_set_v16qi
  // CIR: %[[VEC:.*]] = cir.load
  // CIR: %[[VAL:.*]] = cir.load
  // CIR: %[[IDX:.*]] = cir.const #cir.int<5> : !s64i
  // CIR: %[[RESULT:.*]] = cir.vec.insert %[[VAL]], %[[VEC]][%[[IDX]] : !s64i] : !cir.vector<16 x !s8i>
  // CIR: cir.return %[[RESULT]]

  // LLVM-LABEL: @test_vec_set_v16qi
  // LLVM: %[[VEC:.*]] = load <16 x i8>
  // LLVM: %[[VAL:.*]] = load i8
  // LLVM: %[[RESULT:.*]] = insertelement <16 x i8> %[[VEC]], i8 %[[VAL]], i64 5
  // LLVM: ret <16 x i8> %[[RESULT]]

  // OGCG-LABEL: @test_vec_set_v16qi
  // OGCG: %[[VEC:.*]] = load <16 x i8>
  // OGCG: %[[VAL:.*]] = load i8
  // OGCG: %[[RESULT:.*]] = insertelement <16 x i8> %[[VEC]], i8 %[[VAL]], i64 5
  // OGCG: ret <16 x i8> %[[RESULT]]
  return __builtin_ia32_vec_set_v16qi(a, b, 5);
}

__v8hi test_vec_set_v8hi(__v8hi a, short b) {
  // CIR-LABEL: cir.func{{.*}}@test_vec_set_v8hi
  // CIR: %[[VEC:.*]] = cir.load
  // CIR: %[[VAL:.*]] = cir.load
  // CIR: %[[IDX:.*]] = cir.const #cir.int<3> : !s64i
  // CIR: %[[RESULT:.*]] = cir.vec.insert %[[VAL]], %[[VEC]][%[[IDX]] : !s64i] : !cir.vector<8 x !s16i>
  // CIR: cir.return %[[RESULT]]

  // LLVM-LABEL: @test_vec_set_v8hi
  // LLVM: %[[VEC:.*]] = load <8 x i16>
  // LLVM: %[[VAL:.*]] = load i16
  // LLVM: %[[RESULT:.*]] = insertelement <8 x i16> %[[VEC]], i16 %[[VAL]], i64 3
  // LLVM: ret <8 x i16> %[[RESULT]]

  // OGCG-LABEL: @test_vec_set_v8hi
  // OGCG: %[[VEC:.*]] = load <8 x i16>
  // OGCG: %[[VAL:.*]] = load i16
  // OGCG: %[[RESULT:.*]] = insertelement <8 x i16> %[[VEC]], i16 %[[VAL]], i64 3
  // OGCG: ret <8 x i16> %[[RESULT]]
  return __builtin_ia32_vec_set_v8hi(a, b, 3);
}

__v4si test_vec_set_v4si(__v4si a, int b) {
  // CIR-LABEL: cir.func{{.*}}@test_vec_set_v4si
  // CIR: %[[VEC:.*]] = cir.load
  // CIR: %[[VAL:.*]] = cir.load
  // CIR: %[[IDX:.*]] = cir.const #cir.int<1> : !s64i
  // CIR: %[[RESULT:.*]] = cir.vec.insert %[[VAL]], %[[VEC]][%[[IDX]] : !s64i] : !cir.vector<4 x !s32i>
  // CIR: cir.return %[[RESULT]]

  // LLVM-LABEL: @test_vec_set_v4si
  // LLVM: %[[VEC:.*]] = load <4 x i32>
  // LLVM: %[[VAL:.*]] = load i32
  // LLVM: %[[RESULT:.*]] = insertelement <4 x i32> %[[VEC]], i32 %[[VAL]], i64 1
  // LLVM: ret <4 x i32> %[[RESULT]]

  // OGCG-LABEL: @test_vec_set_v4si
  // OGCG: %[[VEC:.*]] = load <4 x i32>
  // OGCG: %[[VAL:.*]] = load i32
  // OGCG: %[[RESULT:.*]] = insertelement <4 x i32> %[[VEC]], i32 %[[VAL]], i64 1
  // OGCG: ret <4 x i32> %[[RESULT]]
  return __builtin_ia32_vec_set_v4si(a, b, 1);
}

__v2di test_vec_set_v2di(__v2di a, long long b) {
  // CIR-LABEL: cir.func{{.*}}@test_vec_set_v2di
  // CIR: %[[VEC:.*]] = cir.load
  // CIR: %[[VAL:.*]] = cir.load
  // CIR: %[[IDX:.*]] = cir.const #cir.int<0> : !s64i
  // CIR: %[[RESULT:.*]] = cir.vec.insert %[[VAL]], %[[VEC]][%[[IDX]] : !s64i] : !cir.vector<2 x !s64i>
  // CIR: cir.return %[[RESULT]]

  // LLVM-LABEL: @test_vec_set_v2di
  // LLVM: %[[VEC:.*]] = load <2 x i64>
  // LLVM: %[[VAL:.*]] = load i64
  // LLVM: %[[RESULT:.*]] = insertelement <2 x i64> %[[VEC]], i64 %[[VAL]], i64 0
  // LLVM: ret <2 x i64> %[[RESULT]]

  // OGCG-LABEL: @test_vec_set_v2di
  // OGCG: %[[VEC:.*]] = load <2 x i64>
  // OGCG: %[[VAL:.*]] = load i64
  // OGCG: %[[RESULT:.*]] = insertelement <2 x i64> %[[VEC]], i64 %[[VAL]], i64 0
  // OGCG: ret <2 x i64> %[[RESULT]]
  return __builtin_ia32_vec_set_v2di(a, b, 0);
}
