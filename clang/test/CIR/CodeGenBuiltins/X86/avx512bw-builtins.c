// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw  -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512bw -emit-llvm -o - -Wall -Werror -Wsign-conversion | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/avx512bw-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__mmask32 test_kshiftli_mask32(__mmask32 A) {
  // CIR-LABEL: test_kshiftli_mask32
  // CIR: [[VAL:%.*]] = cir.cast bitcast %{{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: [[SHIFT:%.*]] = cir.const #cir.zero : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle([[SHIFT]], [[VAL]] : !cir.vector<32 x !cir.int<s, 1>>) [#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i, #cir.int<32> : !s32i] : !cir.vector<32 x !cir.int<s, 1>>

  // LLVM-LABEL: test_kshiftli_mask32
  // LLVM: [[VAL:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[RES:%.*]] = shufflevector <32 x i1> zeroinitializer, <32 x i1> [[VAL]], <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32>

  // OGCG-LABEL: test_kshiftli_mask32
  // OGCG: [[VAL:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: [[RES:%.*]] = shufflevector <32 x i1> zeroinitializer, <32 x i1> [[VAL]], <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32>
  return _kshiftli_mask32(A, 31);
}

__mmask32 test_kshiftri_mask32(__mmask32 A) {
  // CIR-LABEL: test_kshiftri_mask32
  // CIR: [[VAL:%.*]] = cir.cast bitcast %{{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: [[SHIFT:%.*]] = cir.const #cir.zero : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle([[VAL]], [[SHIFT]] : !cir.vector<32 x !cir.int<s, 1>>) [#cir.int<31> : !s32i, #cir.int<32> : !s32i, #cir.int<33> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<44> : !s32i, #cir.int<45> : !s32i, #cir.int<46> : !s32i, #cir.int<47> : !s32i, #cir.int<48> : !s32i, #cir.int<49> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i, #cir.int<60> : !s32i, #cir.int<61> : !s32i, #cir.int<62> : !s32i] : !cir.vector<32 x !cir.int<s, 1>>

  // LLVM-LABEL: test_kshiftri_mask32
  // LLVM: [[VAL:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[RES:%.*]] = shufflevector <32 x i1> [[VAL]], <32 x i1> zeroinitializer, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>

  // OGCG-LABEL: test_kshiftri_mask32
  // OGCG: [[VAL:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: [[RES:%.*]] = shufflevector <32 x i1> [[VAL]], <32 x i1> zeroinitializer, <32 x i32> <i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62>
  return _kshiftri_mask32(A, 31);
}

__mmask64 test_kshiftli_mask64(__mmask64 A) {
  // CIR-LABEL: test_kshiftli_mask64
  // CIR: [[VAL:%.*]] = cir.cast bitcast %{{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: [[SHIFT:%.*]] = cir.const #cir.zero : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle([[SHIFT]], [[VAL]] : !cir.vector<64 x !cir.int<s, 1>>) [#cir.int<32> : !s32i, #cir.int<33> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<44> : !s32i, #cir.int<45> : !s32i, #cir.int<46> : !s32i, #cir.int<47> : !s32i, #cir.int<48> : !s32i, #cir.int<49> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i, #cir.int<60> : !s32i, #cir.int<61> : !s32i, #cir.int<62> : !s32i, #cir.int<63> : !s32i, #cir.int<64> : !s32i, #cir.int<65> : !s32i, #cir.int<66> : !s32i, #cir.int<67> : !s32i, #cir.int<68> : !s32i, #cir.int<69> : !s32i, #cir.int<70> : !s32i, #cir.int<71> : !s32i, #cir.int<72> : !s32i, #cir.int<73> : !s32i, #cir.int<74> : !s32i, #cir.int<75> : !s32i, #cir.int<76> : !s32i, #cir.int<77> : !s32i, #cir.int<78> : !s32i, #cir.int<79> : !s32i, #cir.int<80> : !s32i, #cir.int<81> : !s32i, #cir.int<82> : !s32i, #cir.int<83> : !s32i, #cir.int<84> : !s32i, #cir.int<85> : !s32i, #cir.int<86> : !s32i, #cir.int<87> : !s32i, #cir.int<88> : !s32i, #cir.int<89> : !s32i, #cir.int<90> : !s32i, #cir.int<91> : !s32i, #cir.int<92> : !s32i, #cir.int<93> : !s32i, #cir.int<94> : !s32i, #cir.int<95> : !s32i] : !cir.vector<64 x !cir.int<s, 1>>

  // LLVM-LABEL: test_kshiftli_mask64
  // LLVM: [[VAL:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[RES:%.*]] = shufflevector <64 x i1> zeroinitializer, <64 x i1> [[VAL]], <64 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>

  // OGCG-LABEL: test_kshiftli_mask64
  // OGCG: [[VAL:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: [[RES:%.*]] = shufflevector <64 x i1> zeroinitializer, <64 x i1> [[VAL]], <64 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>
  return _kshiftli_mask64(A, 32);
}

__mmask64 test_kshiftri_mask64(__mmask64 A) {
  // CIR-LABEL: test_kshiftri_mask64
  // CIR: [[VAL:%.*]] = cir.cast bitcast %{{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: [[SHIFT:%.*]] = cir.const #cir.zero : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle([[VAL]], [[SHIFT]] : !cir.vector<64 x !cir.int<s, 1>>) [#cir.int<32> : !s32i, #cir.int<33> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<44> : !s32i, #cir.int<45> : !s32i, #cir.int<46> : !s32i, #cir.int<47> : !s32i, #cir.int<48> : !s32i, #cir.int<49> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i, #cir.int<60> : !s32i, #cir.int<61> : !s32i, #cir.int<62> : !s32i, #cir.int<63> : !s32i, #cir.int<64> : !s32i, #cir.int<65> : !s32i, #cir.int<66> : !s32i, #cir.int<67> : !s32i, #cir.int<68> : !s32i, #cir.int<69> : !s32i, #cir.int<70> : !s32i, #cir.int<71> : !s32i, #cir.int<72> : !s32i, #cir.int<73> : !s32i, #cir.int<74> : !s32i, #cir.int<75> : !s32i, #cir.int<76> : !s32i, #cir.int<77> : !s32i, #cir.int<78> : !s32i, #cir.int<79> : !s32i, #cir.int<80> : !s32i, #cir.int<81> : !s32i, #cir.int<82> : !s32i, #cir.int<83> : !s32i, #cir.int<84> : !s32i, #cir.int<85> : !s32i, #cir.int<86> : !s32i, #cir.int<87> : !s32i, #cir.int<88> : !s32i, #cir.int<89> : !s32i, #cir.int<90> : !s32i, #cir.int<91> : !s32i, #cir.int<92> : !s32i, #cir.int<93> : !s32i, #cir.int<94> : !s32i, #cir.int<95> : !s32i] : !cir.vector<64 x !cir.int<s, 1>>

  // LLVM-LABEL: test_kshiftri_mask64
  // LLVM: [[VAL:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[RES:%.*]] = shufflevector <64 x i1> [[VAL]], <64 x i1> zeroinitializer, <64 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>

  // OGCG-LABEL: test_kshiftri_mask64
  // OGCG: [[VAL:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: [[RES:%.*]] = shufflevector <64 x i1> [[VAL]], <64 x i1> zeroinitializer, <64 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>
  return _kshiftri_mask64(A, 32);
}

__mmask32 test_kshiftli_mask32_out_of_range(__mmask32 A) {
  // CIR-LABEL: test_kshiftli_mask32_out_of_range
  // CIR: [[VAL:%.*]] = cir.const #cir.int<0> : !u32i
  // CIR: cir.store [[VAL]], {{%.*}} : !u32i, !cir.ptr<!u32i>
  // CIR: [[RES:%.*]] = cir.load {{%.*}} : !cir.ptr<!u32i>, !u32i
  // CIR: cir.return [[RES]] : !u32i

  // LLVM-LABEL: test_kshiftli_mask32_out_of_range
  // LLVM: store i32 0, ptr [[VAL:%.*]], align 4
  // LLVM: [[RES:%.*]] = load i32, ptr [[VAL]], align 4
  // LLVM: ret i32 [[RES]]

  // OGCG-LABEL: test_kshiftli_mask32_out_of_range
  // OGCG: ret i32 0

  return _kshiftli_mask32(A, 33);
}

__mmask32 test_kshiftri_mask32_out_of_range(__mmask32 A) {
  // CIR-LABEL: test_kshiftri_mask32_out_of_range
  // CIR: [[VAL:%.*]] = cir.const #cir.int<0> : !u32i
  // CIR: cir.store [[VAL]], {{%.*}} : !u32i, !cir.ptr<!u32i>
  // CIR: [[RES:%.*]] = cir.load {{%.*}} : !cir.ptr<!u32i>, !u32i
  // CIR: cir.return [[RES]] : !u32i

  // LLVM-LABEL: test_kshiftri_mask32_out_of_range
  // LLVM: store i32 0, ptr [[VAL:%.*]], align 4
  // LLVM: [[RES:%.*]] = load i32, ptr [[VAL]], align 4
  // LLVM: ret i32 [[RES]]

  // OGCG-LABEL: test_kshiftri_mask32_out_of_range
  // OGCG: ret i32 0

  return _kshiftri_mask32(A, 33);
}

__mmask32 test_kadd_mask32(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _kadd_mask32
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.call_llvm_intrinsic "x86.avx512.kadd.d"
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: _kadd_mask32
  // LLVM: [[L:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[R:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[RES:%.*]] = call <32 x i1> @llvm.x86.avx512.kadd.d(<32 x i1> [[L]], <32 x i1> [[R]])
  // LLVM: bitcast <32 x i1> [[RES]] to i32

  // OGCG-LABEL: _kadd_mask32
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: call <32 x i1> @llvm.x86.avx512.kadd.d
  // OGCG: bitcast <32 x i1> {{.*}} to i32
  return _kadd_mask32(A, B);
}

__mmask64 test_kadd_mask64(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _kadd_mask64
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.call_llvm_intrinsic "x86.avx512.kadd.q"
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: _kadd_mask64
  // LLVM: [[L:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[R:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[RES:%.*]] = call <64 x i1> @llvm.x86.avx512.kadd.q(<64 x i1> [[L]], <64 x i1> [[R]])
  // LLVM: bitcast <64 x i1> [[RES]] to i64

  // OGCG-LABEL: _kadd_mask64
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: call <64 x i1> @llvm.x86.avx512.kadd.q
  // OGCG: bitcast <64 x i1> {{.*}} to i64
  return _kadd_mask64(A, B);
}

__mmask32 test_kand_mask32(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _kand_mask32
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: _kand_mask32
  // LLVM: [[L:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[R:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[RES:%.*]] = and <32 x i1> [[L]], [[R]]
  // LLVM: bitcast <32 x i1> [[RES]] to i32

  // OGCG-LABEL: _kand_mask32
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: and <32 x i1>
  // OGCG: bitcast <32 x i1> {{.*}} to i32
  return _kand_mask32(A, B);
}

__mmask64 test_kand_mask64(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _kand_mask64
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: _kand_mask64
  // LLVM: [[L:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[R:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[RES:%.*]] = and <64 x i1> [[L]], [[R]]
  // LLVM: bitcast <64 x i1> [[RES]] to i64

  // OGCG-LABEL: _kand_mask64
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: and <64 x i1>
  // OGCG: bitcast <64 x i1> {{.*}} to i64
  return _kand_mask64(A, B);
}

__mmask32 test_kandn_mask32(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _kandn_mask32
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: _kandn_mask32
  // LLVM: [[L:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[R:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: xor <32 x i1> [[L]], splat (i1 true)
  // LLVM: and <32 x i1>
  // LLVM: bitcast <32 x i1> {{.*}} to i32

  // OGCG-LABEL: _kandn_mask32
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: xor <32 x i1>
  // OGCG: and <32 x i1>
  // OGCG: bitcast <32 x i1> {{.*}} to i32
  return _kandn_mask32(A, B);
}

__mmask64 test_kandn_mask64(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _kandn_mask64
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.binop(and, {{.*}}, {{.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: _kandn_mask64
  // LLVM: [[L:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[R:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: xor <64 x i1> [[L]], splat (i1 true)
  // LLVM: and <64 x i1>
  // LLVM: bitcast <64 x i1> {{.*}} to i64

  // OGCG-LABEL: _kandn_mask64
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: xor <64 x i1>
  // OGCG: and <64 x i1>
  // OGCG: bitcast <64 x i1> {{.*}} to i64
  return _kandn_mask64(A, B);
}

__mmask32 test_kor_mask32(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _kor_mask32
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(or, {{.*}}, {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: _kor_mask32
  // LLVM: [[L:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[R:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: or <32 x i1> [[L]], [[R]]
  // LLVM: bitcast <32 x i1> {{.*}} to i32

  // OGCG-LABEL: _kor_mask32
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: or <32 x i1>
  // OGCG: bitcast <32 x i1> {{.*}} to i32
  return _kor_mask32(A, B);
}

__mmask64 test_kor_mask64(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _kor_mask64
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.binop(or, {{.*}}, {{.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: _kor_mask64
  // LLVM: [[L:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[R:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: or <64 x i1> [[L]], [[R]]
  // LLVM: bitcast <64 x i1> {{.*}} to i64

  // OGCG-LABEL: _kor_mask64
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: or <64 x i1>
  // OGCG: bitcast <64 x i1> {{.*}} to i64
  return _kor_mask64(A, B);
}

__mmask32 test_kxor_mask32(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _kxor_mask32
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: _kxor_mask32
  // LLVM: [[L:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[R:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: xor <32 x i1> [[L]], [[R]]
  // LLVM: bitcast <32 x i1> {{.*}} to i32

  // OGCG-LABEL: _kxor_mask32
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: xor <32 x i1>
  // OGCG: bitcast <32 x i1> {{.*}} to i32
  return _kxor_mask32(A, B);
}

__mmask64 test_kxor_mask64(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _kxor_mask64
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: _kxor_mask64
  // LLVM: [[L:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[R:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: xor <64 x i1> [[L]], [[R]]
  // LLVM: bitcast <64 x i1> {{.*}} to i64

  // OGCG-LABEL: _kxor_mask64
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: xor <64 x i1>
  // OGCG: bitcast <64 x i1> {{.*}} to i64
  return _kxor_mask64(A, B);
}

__mmask32 test_kxnor_mask32(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _kxnor_mask32
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: _kxnor_mask32
  // LLVM: [[L:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[R:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[NOT:%.*]] = xor <32 x i1> [[L]], splat (i1 true)
  // LLVM: [[RES:%.*]] = xor <32 x i1> [[NOT]], [[R]]
  // LLVM: bitcast <32 x i1> [[RES]] to i32

  // OGCG-LABEL: _kxnor_mask32
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: xor <32 x i1>
  // OGCG: xor <32 x i1>
  // OGCG: bitcast <32 x i1> {{.*}} to i32

  return _kxnor_mask32(A, B);
}

__mmask64 test_kxnor_mask64(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _kxnor_mask64
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.binop(xor, {{.*}}, {{.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: _kxnor_mask64
  // LLVM: [[L:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[R:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[NOT:%.*]] = xor <64 x i1> [[L]], splat (i1 true)
  // LLVM: [[RES:%.*]] = xor <64 x i1> [[NOT]], [[R]]
  // LLVM: bitcast <64 x i1> [[RES]] to i64

  // OGCG-LABEL: _kxnor_mask64
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: xor <64 x i1>
  // OGCG: xor <64 x i1>
  // OGCG: bitcast <64 x i1> {{.*}} to i64

  return _kxnor_mask64(A, B);
}


__mmask32 test_knot_mask32(__mmask32 A) {
  // CIR-LABEL: _knot_mask32
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: _knot_mask32
  // LLVM: bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: xor <32 x i1>
  // LLVM: bitcast <32 x i1> {{.*}} to i32

  // OGCG-LABEL: _knot_mask32
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: xor <32 x i1>
  // OGCG: bitcast <32 x i1> {{.*}} to i32
  return _knot_mask32(A);
}

__mmask64 test_knot_mask64(__mmask64 A) {
  // CIR-LABEL: _knot_mask64
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.unary(not, {{.*}}) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: _knot_mask64
  // LLVM: bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: xor <64 x i1>
  // LLVM: bitcast <64 x i1> {{.*}} to i64

  // OGCG-LABEL: _knot_mask64
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: xor <64 x i1>
  // OGCG: bitcast <64 x i1> {{.*}} to i64
  return _knot_mask64(A);
}

// Multiple user-level mask helpers inline to this same kmov builtin.
// CIR does not implement any special lowering for those helpers.
//
// Therefore, testing the builtin (__builtin_ia32_kmov*) directly is
// sufficient to cover the CIR lowering behavior. Testing each helper
// individually would add no new CIR paths.

__mmask32 test_kmov_d(__mmask32 A) {
  // CIR-LABEL: test_kmov_d
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: test_kmov_d
  // LLVM: bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: bitcast <32 x i1> {{.*}} to i32

  // OGCG-LABEL: test_kmov_d
  // OGCG: bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: bitcast <32 x i1> {{.*}} to i32

  return __builtin_ia32_kmovd(A);
}

// Multiple user-level mask helpers inline to this same kmov builtin.
// CIR does not implement any special lowering for those helpers.
//
// Therefore, testing the builtin (__builtin_ia32_kmov*) directly is
// sufficient to cover the CIR lowering behavior. Testing each helper
// individually would add no new CIR paths.

__mmask64 test_kmov_q(__mmask64 A) {
  // CIR-LABEL: test_kmov_q
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: test_kmov_q
  // LLVM: bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: bitcast <64 x i1> {{.*}} to i64

  // OGCG-LABEL: test_kmov_q
  // OGCG: bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: bitcast <64 x i1> {{.*}} to i64

  return __builtin_ia32_kmovq(A);
}

__mmask32 test_mm512_kunpackw(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _mm512_kunpackw
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle
  // CIR: cir.vec.shuffle
  // CIR: cir.vec.shuffle
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: _mm512_kunpackw
  // LLVM: [[A_VEC:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[B_VEC:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: [[A_HALF:%.*]] = shufflevector <32 x i1> [[A_VEC]], <32 x i1> [[A_VEC]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: [[B_HALF:%.*]] = shufflevector <32 x i1> [[B_VEC]], <32 x i1> [[B_VEC]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // LLVM: [[RES:%.*]] = shufflevector <16 x i1> [[B_HALF]], <16 x i1> [[A_HALF]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // LLVM: bitcast <32 x i1> [[RES]] to i32

  // OGCG-LABEL: _mm512_kunpackw
  // OGCG: [[A_VEC:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: [[B_VEC:%.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: [[A_HALF:%.*]] = shufflevector <32 x i1> [[A_VEC]], <32 x i1> [[A_VEC]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // OGCG: [[B_HALF:%.*]] = shufflevector <32 x i1> [[B_VEC]], <32 x i1> [[B_VEC]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // OGCG: [[RES:%.*]] = shufflevector <16 x i1> [[B_HALF]], <16 x i1> [[A_HALF]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // OGCG: bitcast <32 x i1> [[RES]] to i32
  return _mm512_kunpackw(A, B);
}

__mmask64 test_mm512_kunpackd(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _mm512_kunpackd
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: cir.vec.shuffle
  // CIR: cir.vec.shuffle
  // CIR: cir.vec.shuffle
  // CIR: cir.cast bitcast {{.*}} : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: _mm512_kunpackd
  // LLVM: [[A_VEC:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[B_VEC:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: [[A_HALF:%.*]] = shufflevector <64 x i1> [[A_VEC]], <64 x i1> [[A_VEC]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // LLVM: [[B_HALF:%.*]] = shufflevector <64 x i1> [[B_VEC]], <64 x i1> [[B_VEC]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // LLVM: [[RES:%.*]] = shufflevector <32 x i1> [[B_HALF]], <32 x i1> [[A_HALF]], <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  // LLVM: bitcast <64 x i1> [[RES]] to i64

  // OGCG-LABEL: _mm512_kunpackd
  // OGCG: [[A_VEC:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: [[B_VEC:%.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: [[A_HALF:%.*]] = shufflevector <64 x i1> [[A_VEC]], <64 x i1> [[A_VEC]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // OGCG: [[B_HALF:%.*]] = shufflevector <64 x i1> [[B_VEC]], <64 x i1> [[B_VEC]], <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // OGCG: [[RES:%.*]] = shufflevector <32 x i1> [[B_HALF]], <32 x i1> [[A_HALF]], <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  // OGCG: bitcast <64 x i1> [[RES]] to i64
  return _mm512_kunpackd(A, B);
}

__m512i test_mm512_shufflelo_epi16(__m512i __A) {
  // CIR-LABEL: _mm512_shufflelo_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<32 x !s16i>) [#cir.int<1> : !s32i, #cir.int<1> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<9> : !s32i, #cir.int<9> : !s32i, #cir.int<8> : !s32i, #cir.int<8> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<17> : !s32i, #cir.int<17> : !s32i, #cir.int<16> : !s32i, #cir.int<16> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<25> : !s32i, #cir.int<25> : !s32i, #cir.int<24> : !s32i, #cir.int<24> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i] : !cir.vector<32 x !s16i>

  // LLVM-LABEL: test_mm512_shufflelo_epi16
  // LLVM: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>

  // OGCG-LABEL: test_mm512_shufflelo_epi16
  // OGCG: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 1, i32 1, i32 0, i32 0, i32 4, i32 5, i32 6, i32 7, i32 9, i32 9, i32 8, i32 8, i32 12, i32 13, i32 14, i32 15, i32 17, i32 17, i32 16, i32 16, i32 20, i32 21, i32 22, i32 23, i32 25, i32 25, i32 24, i32 24, i32 28, i32 29, i32 30, i32 31>
  return _mm512_shufflelo_epi16(__A, 5);
}

__m512i test_mm512_shufflehi_epi16(__m512i __A) {
  // CIR-LABEL: _mm512_shufflehi_epi16
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<32 x !s16i>) [#cir.int<0> : !s32i, #cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<5> : !s32i, #cir.int<5> : !s32i, #cir.int<4> : !s32i, #cir.int<4> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<13> : !s32i, #cir.int<13> : !s32i, #cir.int<12> : !s32i, #cir.int<12> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<21> : !s32i, #cir.int<21> : !s32i, #cir.int<20> : !s32i, #cir.int<20> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<29> : !s32i, #cir.int<29> : !s32i, #cir.int<28> : !s32i, #cir.int<28> : !s32i] : !cir.vector<32 x !s16i>

  // LLVM-LABEL: test_mm512_shufflehi_epi16
  // LLVM: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>

  // OGCG-LABEL: test_mm512_shufflehi_epi16
  // OGCG: shufflevector <32 x i16> %{{.*}}, <32 x i16> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 5, i32 5, i32 4, i32 4, i32 8, i32 9, i32 10, i32 11, i32 13, i32 13, i32 12, i32 12, i32 16, i32 17, i32 18, i32 19, i32 21, i32 21, i32 20, i32 20, i32 24, i32 25, i32 26, i32 27, i32 29, i32 29, i32 28, i32 28>
  return _mm512_shufflehi_epi16(__A, 5);
}

unsigned char test_kortestc_mask32_u8(__mmask32 __A, __mmask32 __B) {
  // CIR-LABEL: _kortestc_mask32_u8
  // CIR: %[[ALL_ONES:.*]] = cir.const #cir.int<4294967295> : !u32i
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[OR:.*]]  = cir.binop(or, %[[LHS]], %[[RHS]]) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[OR_INT:.*]] = cir.cast bitcast %[[OR]] : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // CIR: %[[CMP:.*]] = cir.cmp(eq, %[[OR_INT]], %[[ALL_ONES]]) : !u32i, !cir.bool
  // CIR: %[[B2I:.*]] = cir.cast bool_to_int %[[CMP]] : !cir.bool -> !s32i
  // CIR: cir.cast integral %[[B2I]] : !s32i -> !u8i

  // LLVM-LABEL: _kortestc_mask32_u8
  // LLVM: %[[LHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[OR:.*]]  = or <32 x i1> %[[LHS]], %[[RHS]]
  // LLVM: %[[CAST:.*]] = bitcast <32 x i1> %[[OR]] to i32
  // LLVM: %[[CMP:.*]] = icmp eq i32 %[[CAST]], -1
  // LLVM: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // LLVM: trunc i32 %[[ZEXT]] to i8

  // OGCG-LABEL: _kortestc_mask32_u8
  // OGCG: %[[LHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[OR:.*]]  = or <32 x i1> %[[LHS]], %[[RHS]]
  // OGCG: %[[CAST:.*]] = bitcast <32 x i1> %[[OR]] to i32
  // OGCG: %[[CMP:.*]] = icmp eq i32 %[[CAST]], -1
  // OGCG: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // OGCG: trunc i32 %[[ZEXT]] to i8
  return _kortestc_mask32_u8(__A, __B);
}

unsigned char test_kortestc_mask64_u8(__mmask64 __A, __mmask64 __B) {
  // CIR-LABEL: _kortestc_mask64_u8
  // CIR: %[[ALL_ONES:.*]] = cir.const #cir.int<18446744073709551615> : !u64i
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[OR:.*]]  = cir.binop(or, %[[LHS]], %[[RHS]]) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[OR_INT:.*]] = cir.cast bitcast %[[OR]] : !cir.vector<64 x !cir.int<s, 1>> -> !u64i
  // CIR: %[[CMP:.*]] = cir.cmp(eq, %[[OR_INT]], %[[ALL_ONES]]) : !u64i, !cir.bool
  // CIR: %[[B2I:.*]] = cir.cast bool_to_int %[[CMP]] : !cir.bool -> !s32i
  // CIR: cir.cast integral %[[B2I]] : !s32i -> !u8i

  // LLVM-LABEL: _kortestc_mask64_u8
  // LLVM: %[[LHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: %[[OR:.*]]  = or <64 x i1> %[[LHS]], %[[RHS]]
  // LLVM: %[[CAST:.*]] = bitcast <64 x i1> %[[OR]] to i64
  // LLVM: %[[CMP:.*]] = icmp eq i64 %[[CAST]], -1
  // LLVM: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // LLVM: trunc i32 %[[ZEXT]] to i8

  // OGCG-LABEL: _kortestc_mask64_u8
  // OGCG: %[[LHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %[[OR:.*]]  = or <64 x i1> %[[LHS]], %[[RHS]]
  // OGCG: %[[CAST:.*]] = bitcast <64 x i1> %[[OR]] to i64
  // OGCG: %[[CMP:.*]] = icmp eq i64 %[[CAST]], -1
  // OGCG: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // OGCG: trunc i32 %[[ZEXT]] to i8
  return _kortestc_mask64_u8(__A, __B);
}

unsigned char test_kortestz_mask32_u8(__mmask32 __A, __mmask32 __B) {
  // CIR-LABEL: _kortestz_mask32_u8
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[OR:.*]]  = cir.binop(or, %[[LHS]], %[[RHS]]) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[OR_INT:.*]] = cir.cast bitcast %[[OR]] : !cir.vector<32 x !cir.int<s, 1>> -> !u32i
  // CIR: %[[CMP:.*]] = cir.cmp(eq, %[[OR_INT]], %[[ZERO]]) : !u32i, !cir.bool
  // CIR: %[[B2I:.*]] = cir.cast bool_to_int %[[CMP]] : !cir.bool -> !s32i
  // CIR: cir.cast integral %[[B2I]] : !s32i -> !u8i

  // LLVM-LABEL: _kortestz_mask32_u8
  // LLVM: %[[LHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[OR:.*]]  = or <32 x i1> %[[LHS]], %[[RHS]]
  // LLVM: %[[CAST:.*]] = bitcast <32 x i1> %[[OR]] to i32
  // LLVM: %[[CMP:.*]] = icmp eq i32 %[[CAST]], 0
  // LLVM: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // LLVM: trunc i32 %[[ZEXT]] to i8

  // OGCG-LABEL: _kortestz_mask32_u8
  // OGCG: %[[LHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[OR:.*]]  = or <32 x i1> %[[LHS]], %[[RHS]]
  // OGCG: %[[CAST:.*]] = bitcast <32 x i1> %[[OR]] to i32
  // OGCG: %[[CMP:.*]] = icmp eq i32 %[[CAST]], 0
  // OGCG: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // OGCG: trunc i32 %[[ZEXT]] to i8
  return _kortestz_mask32_u8(__A, __B);
}

unsigned char test_kortestz_mask64_u8(__mmask64 __A, __mmask64 __B) {
  // CIR-LABEL: _kortestz_mask64_u8
  // CIR: %[[ZERO:.*]] = cir.const #cir.int<0> : !u64i
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[OR:.*]]  = cir.binop(or, %[[LHS]], %[[RHS]]) : !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[OR_INT:.*]] = cir.cast bitcast %[[OR]] : !cir.vector<64 x !cir.int<s, 1>> -> !u64i
  // CIR: %[[CMP:.*]] = cir.cmp(eq, %[[OR_INT]], %[[ZERO]]) : !u64i, !cir.bool
  // CIR: %[[B2I:.*]] = cir.cast bool_to_int %[[CMP]] : !cir.bool -> !s32i
  // CIR: cir.cast integral %[[B2I]] : !s32i -> !u8i

  // LLVM-LABEL: _kortestz_mask64_u8
  // LLVM: %[[LHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: %[[OR:.*]]  = or <64 x i1> %[[LHS]], %[[RHS]]
  // LLVM: %[[CAST:.*]] = bitcast <64 x i1> %[[OR]] to i64
  // LLVM: %[[CMP:.*]] = icmp eq i64 %[[CAST]], 0
  // LLVM: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // LLVM: trunc i32 %[[ZEXT]] to i8

  // OGCG-LABEL: _kortestz_mask64_u8
  // OGCG: %[[LHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %[[OR:.*]]  = or <64 x i1> %[[LHS]], %[[RHS]]
  // OGCG: %[[CAST:.*]] = bitcast <64 x i1> %[[OR]] to i64
  // OGCG: %[[CMP:.*]] = icmp eq i64 %[[CAST]], 0
  // OGCG: %[[ZEXT:.*]] = zext i1 %[[CMP]] to i32
  // OGCG: trunc i32 %[[ZEXT]] to i8
  return _kortestz_mask64_u8(__A, __B);
}

unsigned char test_ktestc_mask32_u8(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _ktestc_mask32_u8
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.ktestc.d"
  // CIR: cir.cast integral %[[RES]] : {{.*}} -> !u8i

  // LLVM-LABEL: _ktestc_mask32_u8
  // LLVM: %[[LHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestc.d(<32 x i1> %[[LHS]], <32 x i1> %[[RHS]])
  // LLVM: trunc i32 %[[RES]] to i8

  // OGCG-LABEL: _ktestc_mask32_u8
  // OGCG: %[[LHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestc.d
  // OGCG: trunc i32 %[[RES]] to i8
  return _ktestc_mask32_u8(A, B);
}

unsigned char test_ktestz_mask32_u8(__mmask32 A, __mmask32 B) {
  // CIR-LABEL: _ktestz_mask32_u8
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.ktestz.d"
  // CIR: cir.cast integral %[[RES]] : {{.*}} -> !u8i

  // LLVM-LABEL: _ktestz_mask32_u8
  // LLVM: %[[LHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestz.d(<32 x i1> %[[LHS]], <32 x i1> %[[RHS]])
  // LLVM: trunc i32 %[[RES]] to i8

  // OGCG-LABEL: _ktestz_mask32_u8
  // OGCG: %[[LHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestz.d
  // OGCG: trunc i32 %[[RES]] to i8
  return _ktestz_mask32_u8(A, B);
}

unsigned char test_ktestc_mask64_u8(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _ktestc_mask64_u8
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.ktestc.q"
  // CIR: cir.cast integral %[[RES]] : {{.*}} -> !u8i

  // LLVM-LABEL: _ktestc_mask64_u8
  // LLVM: %[[LHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestc.q(<64 x i1> %[[LHS]], <64 x i1> %[[RHS]])
  // LLVM: trunc i32 %[[RES]] to i8

  // OGCG-LABEL: _ktestc_mask64_u8
  // OGCG: %[[LHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestc.q
  // OGCG: trunc i32 %[[RES]] to i8
  return _ktestc_mask64_u8(A, B);
}

unsigned char test_ktestz_mask64_u8(__mmask64 A, __mmask64 B) {
  // CIR-LABEL: _ktestz_mask64_u8
  // CIR: %[[LHS:.*]] = cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[RHS:.*]] = cir.cast bitcast {{.*}} : !u64i -> !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %[[RES:.*]] = cir.call_llvm_intrinsic "x86.avx512.ktestz.q"
  // CIR: cir.cast integral %[[RES]] : {{.*}} -> !u8i

  // LLVM-LABEL: _ktestz_mask64_u8
  // LLVM: %[[LHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: %[[RHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // LLVM: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestz.q(<64 x i1> %[[LHS]], <64 x i1> %[[RHS]])
  // LLVM: trunc i32 %[[RES]] to i8

  // OGCG-LABEL: _ktestz_mask64_u8
  // OGCG: %[[LHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %[[RHS:.*]] = bitcast i64 %{{.*}} to <64 x i1>
  // OGCG: %[[RES:.*]] = call i32 @llvm.x86.avx512.ktestz.q
  // OGCG: trunc i32 %[[RES]] to i8
  return _ktestz_mask64_u8(A, B);
}

__m512i test_mm512_movm_epi16(__mmask32 __A) {
  // CIR-LABEL: _mm512_movm_epi16
  // CIR: %{{.*}} = cir.cast bitcast %{{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast integral %{{.*}} : !cir.vector<32 x !cir.int<s, 1>> -> !cir.vector<32 x !s16i>

  // LLVM-LABEL: test_mm512_movm_epi16
  // LLVM:  %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // LLVM:  %{{.*}} = sext <32 x i1> %{{.*}} to <32 x i16>

  // OGCG-LABEL: {{.*}}movm_epi16{{.*}}(
  // OGCG:  %{{.*}} = bitcast i32 %{{.*}} to <32 x i1>
  // OGCG:  %{{.*}} = sext <32 x i1> %{{.*}} to <32 x i16>
  return _mm512_movm_epi16(__A);
}

__mmask64 test_mm512_movepi8_mask(__m512i __A) {
  // CIR-LABEL: _mm512_movepi8_mask
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<64 x !s8i>
  // CIR: [[CMP:%.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<64 x !s8i>, !cir.vector<64 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast [[CMP]] : !cir.vector<64 x !cir.int<s, 1>> -> !u64i

  // LLVM-LABEL: test_mm512_movepi8_mask
  // LLVM: [[CMP:%.*]] = icmp slt <64 x i8> %{{.*}}, zeroinitializer
  // LLVM: bitcast <64 x i1> [[CMP]] to i64

  // OGCG-LABEL: {{.*}}movepi8_mask{{.*}}(
  // OGCG: [[CMP:%.*]] = icmp slt <64 x i8> %{{.*}}, zeroinitializer
  // OGCG: bitcast <64 x i1> [[CMP]] to i64
  return _mm512_movepi8_mask(__A);
}

__mmask32 test_mm512_movepi16_mask(__m512i __A) {
  // CIR-LABEL: _mm512_movepi16_mask
  // CIR: cir.cast bitcast %{{.*}} : !cir.vector<8 x !s64i> -> !cir.vector<32 x !s16i>
  // CIR: [[CMP:%.*]] = cir.vec.cmp(lt, %{{.*}}, %{{.*}}) : !cir.vector<32 x !s16i>, !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %{{.*}} = cir.cast bitcast [[CMP]] : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: test_mm512_movepi16_mask
  // LLVM: [[CMP:%.*]] = icmp slt <32 x i16> %{{.*}}, zeroinitializer
  // LLVM: bitcast <32 x i1> [[CMP]] to i32

  // OGCG-LABEL: {{.*}}movepi16_mask{{.*}}(
  // OGCG: [[CMP:%.*]] = icmp slt <32 x i16> %{{.*}}, zeroinitializer
  // OGCG: bitcast <32 x i1> [[CMP]] to i32
  return _mm512_movepi16_mask(__A);
}

__m512i test_mm512_alignr_epi8(__m512i __A,__m512i __B) {
  // CIR-LABEL: _mm512_alignr_epi8
  // CIR: %{{.*}} = cir.vec.shuffle(%{{.*}}, %{{.*}} : !cir.vector<64 x {{!s8i|!u8i}}>) [#cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<64> : !s32i, #cir.int<65> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i, #cir.int<80> : !s32i, #cir.int<81> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<44> : !s32i, #cir.int<45> : !s32i, #cir.int<46> : !s32i, #cir.int<47> : !s32i, #cir.int<96> : !s32i, #cir.int<97> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i, #cir.int<60> : !s32i, #cir.int<61> : !s32i, #cir.int<62> : !s32i, #cir.int<63> : !s32i, #cir.int<112> : !s32i, #cir.int<113> : !s32i] : !cir.vector<64 x {{!s8i|!u8i}}>

  // LLVM-LABEL: test_mm512_alignr_epi8
  // LLVM: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113>

  // OGCG-LABEL: test_mm512_alignr_epi8
  // OGCG: shufflevector <64 x i8> %{{.*}}, <64 x i8> %{{.*}}, <64 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 80, i32 81, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 96, i32 97, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 112, i32 113>
  return _mm512_alignr_epi8(__A, __B, 2);
}
