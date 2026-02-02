// RUN: %clang_cc1 %s -O0 -ffreestanding -triple=x86_64-unknown-unknown -target-feature +kl -target-feature +widekl -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 %s -O0 -ffreestanding -triple=i386-unknown-unknown -target-feature +kl -target-feature +widekl -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 %s -O0 -ffreestanding -triple=x86_64-unknown-unknown -target-feature +kl -target-feature +widekl -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 %s -O0 -ffreestanding -triple=i386-unknown-unknown -target-feature +kl -target-feature +widekl -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 %s -O0 -ffreestanding -triple=x86_64-unknown-unknown -target-feature +kl -target-feature +widekl -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 %s -O0 -ffreestanding -triple=i386-unknown-unknown -target-feature +kl -target-feature +widekl -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

#include <x86intrin.h>

// CIR: !rec_anon_struct = !cir.record<struct  {!u32i, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>}>
// CIR: !rec_anon_struct1 = !cir.record<struct  {!u32i, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>, !cir.vector<2 x !u64i>}>

unsigned int test_encodekey128_u32(unsigned int htype, __m128i key, void *h) {
  // CIR-LABEL: _mm_encodekey128_u32
  // CIR: %[[H:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR: %[[OUT_PTR:.*]] = cir.cast bitcast %[[H]] : !cir.ptr<!void> -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: %[[CALL:.*]] = cir.call_llvm_intrinsic "x86.encodekey128" {{.*}} : (!u32i, !cir.vector<2 x !s64i>) -> !rec_anon_struct

  // CIR: %[[X0:.*]] = cir.extract_member %[[CALL]][1] : !rec_anon_struct -> !cir.vector<2 x !u64i>
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[P0:.*]] = cir.ptr_stride %[[OUT_PTR]], %[[C0]] : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: cir.store align(1) %[[X0]], %[[P0]] : !cir.vector<2 x !u64i>, !cir.ptr<!cir.vector<2 x !u64i>>

  // CIR: %[[X1:.*]] = cir.extract_member %[[CALL]][2] : !rec_anon_struct -> !cir.vector<2 x !u64i>
  // CIR: %[[C1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[P1:.*]] = cir.ptr_stride %[[OUT_PTR]], %[[C1]] : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: cir.store align(1) %[[X1]], %[[P1]] : !cir.vector<2 x !u64i>, !cir.ptr<!cir.vector<2 x !u64i>>

  // CIR: %[[X2:.*]] = cir.extract_member %[[CALL]][3] : !rec_anon_struct -> !cir.vector<2 x !u64i>
  // CIR: %[[C2:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: %[[P2:.*]] = cir.ptr_stride %[[OUT_PTR]], %[[C2]] : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: cir.store align(1) %[[X2]], %[[P2]] : !cir.vector<2 x !u64i>, !cir.ptr<!cir.vector<2 x !u64i>>

  // CIR: %[[RET_EXT:.*]] = cir.extract_member %[[CALL]][0] : !rec_anon_struct -> !u32i
  // CIR: cir.store %[[RET_EXT]], %[[RET_PTR:.*]] : !u32i, !cir.ptr<!u32i>
  // CIR: %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!u32i>, !u32i
  // CIR: cir.return %[[RET]] : !u32i

  // LLVM-LABEL: test_encodekey128_u32
  // LLVM: %[[CALL:.*]] = call { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32 %{{.*}}, <2 x i64> %{{.*}})

  // LLVM: %[[X0:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 1
  // LLVM: store <2 x i64> %[[X0]], ptr %[[OUT_PTR:.*]], align 1

  // LLVM: %[[X1:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 2
  // LLVM: %[[P1:.*]] = getelementptr <2 x i64>, ptr %[[OUT_PTR]], i{{32|64}} 1
  // LLVM: store <2 x i64> %[[X1]], ptr %[[P1]], align 1

  // LLVM: %[[X2:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 3
  // LLVM: %[[P2:.*]] = getelementptr <2 x i64>, ptr %[[OUT_PTR]], i{{32|64}} 2
  // LLVM: store <2 x i64> %[[X2]], ptr %[[P2]], align 1

  // LLVM: %[[RET_EXT:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 0
  // LLVM: store i32 %[[RET_EXT]], ptr %[[RET_PTR1:.*]], align 4
  // LLVM: %[[RET1:.*]] = load i32, ptr %[[RET_PTR1:.*]], align 4
  // LLVM:  store i32 %[[RET1]], ptr %[[RET_PTR:.*]], align 4
  // LLVM: %[[RET:.*]] = load i32, ptr %[[RET_PTR]], align 4
  // LLVM: ret i32 %[[RET]]

  // OGCG-LABEL: @test_encodekey128_u32
  // OGCG: %[[OUT_PTR:.*]] = load ptr, ptr %__h.addr.i, align {{4|8}}
  // OGCG: %[[CALL:.*]] = call { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey128(i32 %{{.*}}, <2 x i64> %{{.*}})

  // OGCG: %[[X0:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 1
  // OGCG: store <2 x i64> %[[X0]], ptr %[[OUT_PTR]], align 1

  // OGCG: %[[X1:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 2
  // OGCG: %[[P1:.*]] = getelementptr i8, ptr %[[OUT_PTR]], i{{32|64}} 16
  // OGCG: store <2 x i64> %[[X1]], ptr %[[P1]], align 1

  // OGCG: %[[X2:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 3
  // OGCG: %[[P2:.*]] = getelementptr i8, ptr %[[OUT_PTR]], i{{32|64}} 32
  // OGCG: store <2 x i64> %[[X2]], ptr %[[P2]], align 1

  // OGCG: %[[RET:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 0
  // OGCG: ret i32 %[[RET]]
  return _mm_encodekey128_u32(htype, key, h);
}

unsigned int test_encodekey256_u32(unsigned int htype, __m128i key_lo,
                                   __m128i key_hi, void *h) {
  // CIR-LABEL: _mm_encodekey256_u32
  // CIR: %[[H:.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR: %[[OUT_PTR:.*]] = cir.cast bitcast %[[H]] : !cir.ptr<!void> -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: %[[CALL:.*]] = cir.call_llvm_intrinsic "x86.encodekey256" {{.*}} : (!u32i, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>) -> !rec_anon_struct1

  // CIR: %[[X0:.*]] = cir.extract_member %[[CALL]][1] : !rec_anon_struct1 -> !cir.vector<2 x !u64i>
  // CIR: %[[C0:.*]] = cir.const #cir.int<0> : !s32i
  // CIR: %[[P0:.*]] = cir.ptr_stride %[[OUT_PTR]], %[[C0]] : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: cir.store align(1) %[[X0]], %[[P0]] : !cir.vector<2 x !u64i>, !cir.ptr<!cir.vector<2 x !u64i>>

  // CIR: %[[X1:.*]] = cir.extract_member %[[CALL]][2] : !rec_anon_struct1 -> !cir.vector<2 x !u64i>
  // CIR: %[[C1:.*]] = cir.const #cir.int<1> : !s32i
  // CIR: %[[P1:.*]] = cir.ptr_stride %[[OUT_PTR]], %[[C1]] : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: cir.store align(1) %[[X1]], %[[P1]] : !cir.vector<2 x !u64i>, !cir.ptr<!cir.vector<2 x !u64i>>

  // CIR: %[[X2:.*]] = cir.extract_member %[[CALL]][3] : !rec_anon_struct1 -> !cir.vector<2 x !u64i>
  // CIR: %[[C2:.*]] = cir.const #cir.int<2> : !s32i
  // CIR: %[[P2:.*]] = cir.ptr_stride %[[OUT_PTR]], %[[C2]] : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: cir.store align(1) %[[X2]], %[[P2]] : !cir.vector<2 x !u64i>, !cir.ptr<!cir.vector<2 x !u64i>>

  // CIR: %[[X3:.*]] = cir.extract_member %[[CALL]][4] : !rec_anon_struct1 -> !cir.vector<2 x !u64i>
  // CIR: %[[C3:.*]] = cir.const #cir.int<3> : !s32i
  // CIR: %[[P3:.*]] = cir.ptr_stride %[[OUT_PTR]], %[[C3]] : (!cir.ptr<!cir.vector<2 x !u64i>>, !s32i) -> !cir.ptr<!cir.vector<2 x !u64i>>
  // CIR: cir.store align(1) %[[X3]], %[[P3]] : !cir.vector<2 x !u64i>, !cir.ptr<!cir.vector<2 x !u64i>>

  // CIR: %[[RET_EXT:.*]] = cir.extract_member %[[CALL]][0] : !rec_anon_struct1 -> !u32i
  // CIR: cir.store %[[RET_EXT]], %[[RET_PTR:.*]] : !u32i, !cir.ptr<!u32i>
  // CIR: %[[RET:.*]] = cir.load %[[RET_PTR]] : !cir.ptr<!u32i>, !u32i
  // CIR: cir.return %[[RET]] : !u32i

  // LLVM-LABEL: test_encodekey256_u32
  // LLVM: %[[CALL:.*]] = call { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey256(i32 %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  // LLVM: %[[X0:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 1
  // LLVM: store <2 x i64> %[[X0]], ptr %[[OUT_PTR:.*]], align 1

  // LLVM: %[[X1:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 2
  // LLVM: %[[P1:.*]] = getelementptr <2 x i64>, ptr %[[OUT_PTR]], i{{32|64}} 1
  // LLVM: store <2 x i64> %[[X1]], ptr %[[P1]], align 1

  // LLVM: %[[X2:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 3
  // LLVM: %[[P2:.*]] = getelementptr <2 x i64>, ptr %[[OUT_PTR]], i{{32|64}} 2
  // LLVM: store <2 x i64> %[[X2]], ptr %[[P2]], align 1

  // LLVM: %[[X3:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 4
  // LLVM: %[[P3:.*]] = getelementptr <2 x i64>, ptr %[[OUT_PTR]], i{{32|64}} 3
  // LLVM: store <2 x i64> %[[X3]], ptr %[[P3]], align 1

  // LLVM: %[[RET_EXT:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 0
  // LLVM: store i32 %[[RET_EXT]], ptr [[RET_PTR1:.*]], align 4
  // LLVM: %[[RET1:.*]] = load i32, ptr %[[RET_PTR1:.*]], align 4
  // LLVM:  store i32 %[[RET1]], ptr %[[RET_PTR:.*]], align 4
  // LLVM: %[[RET:.*]] = load i32, ptr %[[RET_PTR]], align 4
  // LLVM: ret i32 %[[RET]]

  // OGCG-LABEL: @test_encodekey256_u32
  // OGCG: %[[OUT_PTR:.*]] = load ptr, ptr %__h.addr.i, align {{4|8|16}}
  // OGCG: %[[CALL:.*]] = call { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.encodekey256(i32 %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})

  // OGCG: %[[X0:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 1
  // OGCG: store <2 x i64> %[[X0]], ptr %[[OUT_PTR]], align 1

  // OGCG: %[[X1:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 2
  // OGCG: %[[P1:.*]] = getelementptr i8, ptr %[[OUT_PTR]], i{{32|64}} 16
  // OGCG: store <2 x i64> %[[X1]], ptr %[[P1]], align 1

  // OGCG: %[[X2:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 3
  // OGCG: %[[P2:.*]] = getelementptr i8, ptr %[[OUT_PTR]], i{{32|64}} 32
  // OGCG: store <2 x i64> %[[X2]], ptr %[[P2]], align 1

  // OGCG: %[[X3:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 4
  // OGCG: %[[P3:.*]] = getelementptr i8, ptr %[[OUT_PTR]], i{{32|64}} 48
  // OGCG: store <2 x i64> %[[X3]], ptr %[[P3]], align 1

  // OGCG: %[[RET_EXT:.*]] = extractvalue { i32, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[CALL]], 0
  // OGCG: ret i32 %[[RET_EXT]]

  return _mm_encodekey256_u32(htype, key_lo, key_hi, h);
}
