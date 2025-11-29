// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -fms-extensions -fms-compatibility -ffreestanding %s -triple=x86_64-windows-msvc -target-feature +avx512f -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

#include <immintrin.h>

__m512 test_mm512_undefined(void) {
  // CIR-LABEL: _mm512_undefined
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<16 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.float>

  // LLVM-LABEL: test_mm512_undefined
  // LLVM: store <16 x float> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <16 x float>, ptr %[[A]], align 64
  // LLVM: ret <16 x float> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined
  // OGCG: ret <16 x float> zeroinitializer
  return _mm512_undefined();
}

__m512 test_mm512_undefined_ps(void) {
  // CIR-LABEL: _mm512_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<16 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<16 x !cir.float>

  // LLVM-LABEL: test_mm512_undefined_ps
  // LLVM: store <16 x float> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <16 x float>, ptr %[[A]], align 64
  // LLVM: ret <16 x float> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_ps
  // OGCG: ret <16 x float> zeroinitializer
  return _mm512_undefined_ps();
}

__m512d test_mm512_undefined_pd(void) {
  // CIR-LABEL: _mm512_undefined_pd
  // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.double>

  // LLVM-LABEL: test_mm512_undefined_pd
  // LLVM: store <8 x double> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <8 x double>, ptr %[[A]], align 64
  // LLVM: ret <8 x double> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_pd
  // OGCG: ret <8 x double> zeroinitializer
  return _mm512_undefined_pd();
}

__m512i test_mm512_undefined_epi32(void) {
  // CIR-LABEL: _mm512_undefined_epi32
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<8 x !s64i>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !s64i>

  // LLVM-LABEL: test_mm512_undefined_epi32
  // LLVM: store <8 x i64> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <8 x i64>, ptr %[[A]], align 64
  // LLVM: ret <8 x i64> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_epi32
  // OGCG: ret <8 x i64> zeroinitializer
  return _mm512_undefined_epi32();
}

__mmask16 test_mm512_kand(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kand
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kand
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[RES:%.*]] = and <16 x i1> [[L]], [[R]]
  // LLVM: bitcast <16 x i1> [[RES]] to i16

  // OGCG-LABEL: _mm512_kand
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: and <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kand(A, B);
}

__mmask16 test_mm512_kandn(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kandn
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kandn
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: xor <16 x i1> [[L]], splat (i1 true)
  // LLVM: and <16 x i1>
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: _mm512_kandn
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: and <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kandn(A, B);
}

__mmask16 test_mm512_kor(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kor
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.binop(or, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kor
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: or <16 x i1> [[L]], [[R]]
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: _mm512_kor
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: or <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kor(A, B);
}

__mmask16 test_mm512_kxnor(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kxnor
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kxnor
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[NOT:%.*]] = xor <16 x i1> [[L]], splat (i1 true)
  // LLVM: [[RES:%.*]] = xor <16 x i1> [[NOT]], [[R]]
  // LLVM: bitcast <16 x i1> [[RES]] to i16

  // OGCG-LABEL: _mm512_kxnor
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kxnor(A, B);
}

__mmask16 test_mm512_kxor(__mmask16 A, __mmask16 B) {
  // CIR-LABEL: _mm512_kxor
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

  // LLVM-LABEL: _mm512_kxor
  // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: xor <16 x i1> [[L]], [[R]]
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: _mm512_kxor
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_kxor(A, B);
}

__mmask16 test_mm512_knot(__mmask16 A) {
  // CIR-LABEL: _mm512_knot
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

  // LLVM-LABEL: _mm512_knot
  // LLVM: bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: xor <16 x i1>
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: _mm512_knot
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: xor <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return _mm512_knot(A);
}

// Multiple user-level mask helpers inline to this same kmov builtin.
// CIR does not implement any special lowering for those helpers.
//
// Therefore, testing the builtin (__builtin_ia32_kmov*) directly is
// sufficient to cover the CIR lowering behavior. Testing each helper
// individually would add no new CIR paths.

__mmask16 test_kmov_w(__mmask16 A) {
  // CIR-LABEL: test_kmov_w
  // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

  // LLVM-LABEL: test_kmov_w
  // LLVM: bitcast i16 %{{.*}} to <16 x i1>
  // LLVM: bitcast <16 x i1> {{.*}} to i16

  // OGCG-LABEL: test_kmov_w
  // OGCG: bitcast i16 %{{.*}} to <16 x i1>
  // OGCG: bitcast <16 x i1> {{.*}} to i16
  return __builtin_ia32_kmovw(A);
}
