// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512bw -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

// This test mimics clang/test/CodeGen/X86/avx512bw-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__mmask32 test_kshiftli_mask32(__mmask32 A) {
  // CIR-LABEL: test_kshiftli_mask32
  // CIR: [[VAL:%.*]] = cir.cast bitcast %{{.*}} : !u32i -> !cir.vector<32 x !cir.int<u, 1>>
  // CIR: [[SHIFT:%.*]] = cir.const #cir.zero : !cir.vector<32 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle([[SHIFT]], [[VAL]] : !cir.vector<32 x !cir.int<u, 1>>) [#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i, #cir.int<4> : !s32i, #cir.int<5> : !s32i, #cir.int<6> : !s32i, #cir.int<7> : !s32i, #cir.int<8> : !s32i, #cir.int<9> : !s32i, #cir.int<10> : !s32i, #cir.int<11> : !s32i, #cir.int<12> : !s32i, #cir.int<13> : !s32i, #cir.int<14> : !s32i, #cir.int<15> : !s32i, #cir.int<16> : !s32i, #cir.int<17> : !s32i, #cir.int<18> : !s32i, #cir.int<19> : !s32i, #cir.int<20> : !s32i, #cir.int<21> : !s32i, #cir.int<22> : !s32i, #cir.int<23> : !s32i, #cir.int<24> : !s32i, #cir.int<25> : !s32i, #cir.int<26> : !s32i, #cir.int<27> : !s32i, #cir.int<28> : !s32i, #cir.int<29> : !s32i, #cir.int<30> : !s32i, #cir.int<31> : !s32i, #cir.int<32> : !s32i] : !cir.vector<32 x !cir.int<u, 1>>

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
  // CIR: [[VAL:%.*]] = cir.cast bitcast %{{.*}} : !u32i -> !cir.vector<32 x !cir.int<u, 1>>
  // CIR: [[SHIFT:%.*]] = cir.const #cir.zero : !cir.vector<32 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle([[VAL]], [[SHIFT]] : !cir.vector<32 x !cir.int<u, 1>>) [#cir.int<31> : !s32i, #cir.int<32> : !s32i, #cir.int<33> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<44> : !s32i, #cir.int<45> : !s32i, #cir.int<46> : !s32i, #cir.int<47> : !s32i, #cir.int<48> : !s32i, #cir.int<49> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i, #cir.int<60> : !s32i, #cir.int<61> : !s32i, #cir.int<62> : !s32i] : !cir.vector<32 x !cir.int<u, 1>>

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
  // CIR: [[VAL:%.*]] = cir.cast bitcast %{{.*}} : !u64i -> !cir.vector<64 x !cir.int<u, 1>>
  // CIR: [[SHIFT:%.*]] = cir.const #cir.zero : !cir.vector<64 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle([[SHIFT]], [[VAL]] : !cir.vector<64 x !cir.int<u, 1>>) [#cir.int<32> : !s32i, #cir.int<33> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<44> : !s32i, #cir.int<45> : !s32i, #cir.int<46> : !s32i, #cir.int<47> : !s32i, #cir.int<48> : !s32i, #cir.int<49> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i, #cir.int<60> : !s32i, #cir.int<61> : !s32i, #cir.int<62> : !s32i, #cir.int<63> : !s32i, #cir.int<64> : !s32i, #cir.int<65> : !s32i, #cir.int<66> : !s32i, #cir.int<67> : !s32i, #cir.int<68> : !s32i, #cir.int<69> : !s32i, #cir.int<70> : !s32i, #cir.int<71> : !s32i, #cir.int<72> : !s32i, #cir.int<73> : !s32i, #cir.int<74> : !s32i, #cir.int<75> : !s32i, #cir.int<76> : !s32i, #cir.int<77> : !s32i, #cir.int<78> : !s32i, #cir.int<79> : !s32i, #cir.int<80> : !s32i, #cir.int<81> : !s32i, #cir.int<82> : !s32i, #cir.int<83> : !s32i, #cir.int<84> : !s32i, #cir.int<85> : !s32i, #cir.int<86> : !s32i, #cir.int<87> : !s32i, #cir.int<88> : !s32i, #cir.int<89> : !s32i, #cir.int<90> : !s32i, #cir.int<91> : !s32i, #cir.int<92> : !s32i, #cir.int<93> : !s32i, #cir.int<94> : !s32i, #cir.int<95> : !s32i] : !cir.vector<64 x !cir.int<u, 1>>

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
  // CIR: [[VAL:%.*]] = cir.cast bitcast %{{.*}} : !u64i -> !cir.vector<64 x !cir.int<u, 1>>
  // CIR: [[SHIFT:%.*]] = cir.const #cir.zero : !cir.vector<64 x !cir.int<u, 1>>
  // CIR: %{{.*}} = cir.vec.shuffle([[VAL]], [[SHIFT]] : !cir.vector<64 x !cir.int<u, 1>>) [#cir.int<32> : !s32i, #cir.int<33> : !s32i, #cir.int<34> : !s32i, #cir.int<35> : !s32i, #cir.int<36> : !s32i, #cir.int<37> : !s32i, #cir.int<38> : !s32i, #cir.int<39> : !s32i, #cir.int<40> : !s32i, #cir.int<41> : !s32i, #cir.int<42> : !s32i, #cir.int<43> : !s32i, #cir.int<44> : !s32i, #cir.int<45> : !s32i, #cir.int<46> : !s32i, #cir.int<47> : !s32i, #cir.int<48> : !s32i, #cir.int<49> : !s32i, #cir.int<50> : !s32i, #cir.int<51> : !s32i, #cir.int<52> : !s32i, #cir.int<53> : !s32i, #cir.int<54> : !s32i, #cir.int<55> : !s32i, #cir.int<56> : !s32i, #cir.int<57> : !s32i, #cir.int<58> : !s32i, #cir.int<59> : !s32i, #cir.int<60> : !s32i, #cir.int<61> : !s32i, #cir.int<62> : !s32i, #cir.int<63> : !s32i, #cir.int<64> : !s32i, #cir.int<65> : !s32i, #cir.int<66> : !s32i, #cir.int<67> : !s32i, #cir.int<68> : !s32i, #cir.int<69> : !s32i, #cir.int<70> : !s32i, #cir.int<71> : !s32i, #cir.int<72> : !s32i, #cir.int<73> : !s32i, #cir.int<74> : !s32i, #cir.int<75> : !s32i, #cir.int<76> : !s32i, #cir.int<77> : !s32i, #cir.int<78> : !s32i, #cir.int<79> : !s32i, #cir.int<80> : !s32i, #cir.int<81> : !s32i, #cir.int<82> : !s32i, #cir.int<83> : !s32i, #cir.int<84> : !s32i, #cir.int<85> : !s32i, #cir.int<86> : !s32i, #cir.int<87> : !s32i, #cir.int<88> : !s32i, #cir.int<89> : !s32i, #cir.int<90> : !s32i, #cir.int<91> : !s32i, #cir.int<92> : !s32i, #cir.int<93> : !s32i, #cir.int<94> : !s32i, #cir.int<95> : !s32i] : !cir.vector<64 x !cir.int<u, 1>>

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
