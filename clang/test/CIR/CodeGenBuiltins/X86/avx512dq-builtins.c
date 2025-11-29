// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512dq -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

__mmask8 test_kadd_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kadd_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.call_llvm_intrinsic "x86.avx512.kadd.b"
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kadd_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[RES:%.*]] = call <8 x i1> @llvm.x86.avx512.kadd.b(<8 x i1> [[L]], <8 x i1> [[R]])
 // LLVM: bitcast <8 x i1> [[RES]] to i8

 // OGCG-LABEL: _kadd_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: call <8 x i1> @llvm.x86.avx512.kadd.b
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kadd_mask8(A, B);
}

__mmask16 test_kadd_mask16(__mmask16 A, __mmask16 B) {
 // CIR-LABEL: _kadd_mask16
 // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u16i -> !cir.vector<16 x !cir.int<u, 1>>
 // CIR: cir.call_llvm_intrinsic "x86.avx512.kadd.w"
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<16 x !cir.int<u, 1>> -> !u16i

 // LLVM-LABEL: _kadd_mask16
 // LLVM: [[L:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
 // LLVM: [[R:%.*]] = bitcast i16 %{{.*}} to <16 x i1>
 // LLVM: [[RES:%.*]] = call <16 x i1> @llvm.x86.avx512.kadd.w(<16 x i1> [[L]], <16 x i1> [[R]])
 // LLVM: bitcast <16 x i1> [[RES]] to i16

 // OGCG-LABEL: _kadd_mask16
 // OGCG: bitcast i16 %{{.*}} to <16 x i1>
 // OGCG: bitcast i16 %{{.*}} to <16 x i1>
 // OGCG: call <16 x i1> @llvm.x86.avx512.kadd.w
 // OGCG: bitcast <16 x i1> {{.*}} to i16
 return _kadd_mask16(A, B);
}

__mmask8 test_kand_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kand_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kand_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[RES:%.*]] = and <8 x i1> [[L]], [[R]]
 // LLVM: bitcast <8 x i1> [[RES]] to i8

 // OGCG-LABEL: _kand_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: and <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kand_mask8(A, B);
}


__mmask8 test_kandn_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kandn_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.unary(not, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kandn_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: xor <8 x i1> [[L]], splat (i1 true)
 // LLVM: and <8 x i1>
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: _kandn_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: and <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8

 return _kandn_mask8(A, B);
}

__mmask8 test_kor_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kor_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(or, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kor_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: or <8 x i1> [[L]], [[R]]
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: _kor_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: or <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kor_mask8(A, B);
}

__mmask8 test_kxor_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kxor_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kxor_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: xor <8 x i1> [[L]], [[R]]
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: _kxor_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kxor_mask8(A, B);
}

__mmask8 test_kxnor_mask8(__mmask8 A, __mmask8 B) {
 // CIR-LABEL: _kxnor_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.unary(not, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _kxnor_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[R:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: [[NOT:%.*]] = xor <8 x i1> [[L]], splat (i1 true)
 // LLVM: [[RES:%.*]] = xor <8 x i1> [[NOT]], [[R]]
 // LLVM: bitcast <8 x i1> [[RES]] to i8

 // OGCG-LABEL: _kxnor_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _kxnor_mask8(A, B);
}


__mmask8 test_knot_mask8(__mmask8 A) {
 // CIR-LABEL: _knot_mask8
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.unary(not, {{.*}}) : !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: _knot_mask8
 // LLVM: [[L:%.*]] = bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: xor <8 x i1> [[L]], {{.*}}
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: _knot_mask8
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: xor <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return _knot_mask8(A);
}

// Multiple user-level mask helpers inline to this same kmov builtin.
// CIR does not implement any special lowering for those helpers.
//
// Therefore, testing the builtin (__builtin_ia32_kmov*) directly is
// sufficient to cover the CIR lowering behavior. Testing each helper
// individually would add no new CIR paths.

__mmask8 test_kmov_b(__mmask8 A) {
 // CIR-LABEL: test_kmov_b
 // CIR: cir.cast bitcast {{.*}} : !u8i -> !cir.vector<8 x !cir.int<u, 1>>
 // CIR: cir.cast bitcast {{.*}} : !cir.vector<8 x !cir.int<u, 1>> -> !u8i

 // LLVM-LABEL: test_kmov_b
 // LLVM: bitcast i8 %{{.*}} to <8 x i1>
 // LLVM: bitcast <8 x i1> {{.*}} to i8

 // OGCG-LABEL: test_kmov_b
 // OGCG: bitcast i8 %{{.*}} to <8 x i1>
 // OGCG: bitcast <8 x i1> {{.*}} to i8
 return __builtin_ia32_kmovb(A);
}
