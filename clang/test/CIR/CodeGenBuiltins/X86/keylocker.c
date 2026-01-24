// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +kl -target-feature +widekl -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +kl -target-feature +widekl -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +kl -target-feature +widekl -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -triple=x86_64-unknown-linux -target-feature +kl -target-feature +widekl -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -ffreestanding -triple=x86_64-unknown-linux -target-feature +kl -target-feature +widekl -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

// This test mimics clang/test/CodeGen/X86/keylocker.c, which eventually
// CIR shall be able to support fully.

#include <x86intrin.h>

// CIR: !rec_anon_struct = !cir.record<struct  {!u8i, !cir.vector<2 x !s64i>}>
// CIR: !rec_anon_struct1 = !cir.record<struct  {!u8i, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>}>

unsigned char test_mm_aesenc256kl_u8(__m128i *odata, __m128i idata, const void *h) {
  // CIR-LABEL: _mm_aesenc256kl_u8
  // CIR:  %[[RESULT:.+]] = cir.call_llvm_intrinsic "x86.aesenc256kl" %[[IDATA:.+]], %[[H:.+]] : (!cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !rec_anon_struct
  // CIR:  %[[FLAG:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct -> !u8i
  // CIR:  %[[FLAG_BIT0:.+]] = cir.cast integral %[[FLAG]] : !u8i -> !cir.int<u, 1>
  // CIR:  %[[SUCC:.+]] = cir.cast int_to_bool %[[FLAG_BIT0]] : !cir.int<u, 1> -> !cir.bool
  // CIR:  %[[OUT:.+]] = cir.extract_member %[[RESULT]][1] : !rec_anon_struct -> !cir.vector<2 x !s64i>
  // CIR:  cir.if %[[SUCC]] {
  // CIR:    cir.store align(16) %[[OUT]], %[[ODATA_PTR:.+]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  } else {
  // CIR:    %[[NULL:.+]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
  // CIR:    cir.store align(16) %[[NULL]], %[[ODATA_PTR]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  }
  // CIR:  %[[FLAG1:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct -> !u8i
  // CIR:  cir.store %[[FLAG1]], %[[RET_ADDR:.+]] : !u8i, !cir.ptr<!u8i>
  // CIR:  %[[RET:.+]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u8i>, !u8i
  // CIR:  cir.return %[[RET]] : !u8i

  // LLVM-LABEL: _mm_aesenc256kl_u8
  // LLVM:   %[[RESULT:.+]] = call { i8, <2 x i64> } @llvm.x86.aesenc256kl(<2 x i64> %[[IDATA:.+]], ptr %[[H:.+]])
  // LLVM:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // LLVM:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // LLVM:   %[[OUT:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 1
  // LLVM:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // LLVM: [[NO_ERROR]]:
  // LLVM:   store <2 x i64> %[[OUT]], ptr %[[ODATA_PTR:.+]], align 16
  // LLVM:   br label %[[EXIT:.+]]
  // LLVM: [[ERROR]]:
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // LLVM:   br label %[[EXIT]]
  // LLVM: [[EXIT]]:
  // LLVM:   %[[FLAG1:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // LLVM:   store i8 %[[FLAG1]], ptr %[[FLAG1_ADDR:.+]], align 1
  // LLVM:   %[[FLAG2:.+]] = load i8, ptr %[[FLAG1_ADDR]], align 1
  // LLVM:   store i8 %[[FLAG2:.+]], ptr %[[FLAG2_ADDR:.+]], align 1
  // LLVM:   %[[RET:.+]] = load i8, ptr %[[FLAG2_ADDR]], align 1
  // LLVM:   ret i8 %[[RET]]

  // OGCG-LABEL: test_mm_aesenc256kl_u8
  // OGCG:   %[[RESULT:.+]] = call { i8, <2 x i64> } @llvm.x86.aesenc256kl(<2 x i64> %[[IDATA:.+]], ptr %[[H:.+]])
  // OGCG:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // OGCG:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // OGCG:   %[[OUT:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 1
  // OGCG:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // OGCG: [[NO_ERROR]]:
  // OGCG:   store <2 x i64> %[[OUT]], ptr %[[ODATA_PTR:.+]], align 16
  // OGCG:   br label %[[EXIT:.+]]
  // OGCG: [[ERROR]]:
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // OGCG:   br label %[[EXIT]]
  // OGCG: [[EXIT]]:
  // OGCG:   %[[RET:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // OGCG:   ret i8 %[[RET]]
  return _mm_aesenc256kl_u8(odata, idata, h);
}

unsigned char test_mm_aesdec256kl_u8(__m128i *odata, __m128i idata, const void *h) {
  // CIR-LABEL: _mm_aesdec256kl_u8
  // CIR:  %[[RESULT:.+]] = cir.call_llvm_intrinsic "x86.aesdec256kl" %[[IDATA:.+]], %[[H:.+]] : (!cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !rec_anon_struct
  // CIR:  %[[FLAG:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct -> !u8i
  // CIR:  %[[FLAG_BIT0:.+]] = cir.cast integral %[[FLAG]] : !u8i -> !cir.int<u, 1>
  // CIR:  %[[SUCC:.+]] = cir.cast int_to_bool %[[FLAG_BIT0]] : !cir.int<u, 1> -> !cir.bool
  // CIR:  %[[OUT:.+]] = cir.extract_member %[[RESULT]][1] : !rec_anon_struct -> !cir.vector<2 x !s64i>
  // CIR:  cir.if %[[SUCC]] {
  // CIR:    cir.store align(16) %[[OUT]], %[[ODATA_PTR:.+]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  } else {
  // CIR:    %[[NULL:.+]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
  // CIR:    cir.store align(16) %[[NULL]], %[[ODATA_PTR]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  }
  // CIR:  %[[FLAG1:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct -> !u8i
  // CIR:  cir.store %[[FLAG1]], %[[RET_ADDR:.+]] : !u8i, !cir.ptr<!u8i>
  // CIR:  %[[RET:.+]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u8i>, !u8i
  // CIR:  cir.return %[[RET]] : !u8i

  // LLVM-LABEL: _mm_aesdec256kl_u8
  // LLVM:   %[[RESULT:.+]] = call { i8, <2 x i64> } @llvm.x86.aesdec256kl(<2 x i64> %[[IDATA:.+]], ptr %[[H:.+]])
  // LLVM:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // LLVM:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // LLVM:   %[[OUT:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 1
  // LLVM:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // LLVM: [[NO_ERROR]]:
  // LLVM:   store <2 x i64> %[[OUT]], ptr %[[ODATA_PTR:.+]], align 16
  // LLVM:   br label %[[EXIT:.+]]
  // LLVM: [[ERROR]]:
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // LLVM:   br label %[[EXIT]]
  // LLVM: [[EXIT]]:
  // LLVM:   %[[FLAG1:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // LLVM:   store i8 %[[FLAG1]], ptr %[[FLAG1_ADDR:.+]], align 1
  // LLVM:   %[[FLAG2:.+]] = load i8, ptr %[[FLAG1_ADDR]], align 1
  // LLVM:   store i8 %[[FLAG2:.+]], ptr %[[FLAG2_ADDR:.+]], align 1
  // LLVM:   %[[RET:.+]] = load i8, ptr %[[FLAG2_ADDR]], align 1
  // LLVM:   ret i8 %[[RET]]

  // OGCG-LABEL: test_mm_aesdec256kl_u8
  // OGCG:   %[[RESULT:.+]] = call { i8, <2 x i64> } @llvm.x86.aesdec256kl(<2 x i64> %[[IDATA:.+]], ptr %[[H:.+]])
  // OGCG:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // OGCG:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // OGCG:   %[[OUT:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 1
  // OGCG:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // OGCG: [[NO_ERROR]]:
  // OGCG:   store <2 x i64> %[[OUT]], ptr %[[ODATA_PTR:.+]], align 16
  // OGCG:   br label %[[EXIT:.+]]
  // OGCG: [[ERROR]]:
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // OGCG:   br label %[[EXIT]]
  // OGCG: [[EXIT]]:
  // OGCG:   %[[RET:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // OGCG:   ret i8 %[[RET]]
  return _mm_aesdec256kl_u8(odata, idata, h);
}

unsigned char test_mm_aesenc128kl_u8(__m128i *odata, __m128i idata, const void *h) {
  // CIR-LABEL: _mm_aesenc128kl_u8
  // CIR:  %[[RESULT:.+]] = cir.call_llvm_intrinsic "x86.aesenc128kl" %[[IDATA:.+]], %[[H:.+]] : (!cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !rec_anon_struct
  // CIR:  %[[FLAG:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct -> !u8i
  // CIR:  %[[FLAG_BIT0:.+]] = cir.cast integral %[[FLAG]] : !u8i -> !cir.int<u, 1>
  // CIR:  %[[SUCC:.+]] = cir.cast int_to_bool %[[FLAG_BIT0]] : !cir.int<u, 1> -> !cir.bool
  // CIR:  %[[OUT:.+]] = cir.extract_member %[[RESULT]][1] : !rec_anon_struct -> !cir.vector<2 x !s64i>
  // CIR:  cir.if %[[SUCC]] {
  // CIR:    cir.store align(16) %[[OUT]], %[[ODATA_PTR:.+]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  } else {
  // CIR:    %[[NULL:.+]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
  // CIR:    cir.store align(16) %[[NULL]], %[[ODATA_PTR]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  }
  // CIR:  %[[FLAG1:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct -> !u8i
  // CIR:  cir.store %[[FLAG1]], %[[RET_ADDR:.+]] : !u8i, !cir.ptr<!u8i>
  // CIR:  %[[RET:.+]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u8i>, !u8i
  // CIR:  cir.return %[[RET]] : !u8i

  // LLVM-LABEL: _mm_aesenc128kl_u8
  // LLVM:   %[[RESULT:.+]] = call { i8, <2 x i64> } @llvm.x86.aesenc128kl(<2 x i64> %[[IDATA:.+]], ptr %[[H:.+]])
  // LLVM:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // LLVM:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // LLVM:   %[[OUT:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 1
  // LLVM:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // LLVM: [[NO_ERROR]]:
  // LLVM:   store <2 x i64> %[[OUT]], ptr %[[ODATA_PTR:.+]], align 16
  // LLVM:   br label %[[EXIT:.+]]
  // LLVM: [[ERROR]]:
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // LLVM:   br label %[[EXIT]]
  // LLVM: [[EXIT]]:
  // LLVM:   %[[FLAG1:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // LLVM:   store i8 %[[FLAG1]], ptr %[[FLAG1_ADDR:.+]], align 1
  // LLVM:   %[[FLAG2:.+]] = load i8, ptr %[[FLAG1_ADDR]], align 1
  // LLVM:   store i8 %[[FLAG2:.+]], ptr %[[FLAG2_ADDR:.+]], align 1
  // LLVM:   %[[RET:.+]] = load i8, ptr %[[FLAG2_ADDR]], align 1
  // LLVM:   ret i8 %[[RET]]

  // OGCG-LABEL: test_mm_aesenc128kl_u8
  // OGCG:   %[[RESULT:.+]] = call { i8, <2 x i64> } @llvm.x86.aesenc128kl(<2 x i64> %[[IDATA:.+]], ptr %[[H:.+]])
  // OGCG:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // OGCG:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // OGCG:   %[[OUT:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 1
  // OGCG:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // OGCG: [[NO_ERROR]]:
  // OGCG:   store <2 x i64> %[[OUT]], ptr %[[ODATA_PTR:.+]], align 16
  // OGCG:   br label %[[EXIT:.+]]
  // OGCG: [[ERROR]]:
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // OGCG:   br label %[[EXIT]]
  // OGCG: [[EXIT]]:
  // OGCG:   %[[RET:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // OGCG:   ret i8 %[[RET]]
  return _mm_aesenc128kl_u8(odata, idata, h);
}

unsigned char test_mm_aesdec128kl_u8(__m128i *odata, __m128i idata, const void *h) {
  // CIR-LABEL: _mm_aesdec128kl_u8
  // CIR:  %[[RESULT:.+]] = cir.call_llvm_intrinsic "x86.aesdec128kl" %[[IDATA:.+]], %[[H:.+]] : (!cir.vector<2 x !s64i>, !cir.ptr<!void>) -> !rec_anon_struct
  // CIR:  %[[FLAG:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct -> !u8i
  // CIR:  %[[FLAG_BIT0:.+]] = cir.cast integral %[[FLAG]] : !u8i -> !cir.int<u, 1>
  // CIR:  %[[SUCC:.+]] = cir.cast int_to_bool %[[FLAG_BIT0]] : !cir.int<u, 1> -> !cir.bool
  // CIR:  %[[OUT:.+]] = cir.extract_member %[[RESULT]][1] : !rec_anon_struct -> !cir.vector<2 x !s64i>
  // CIR:  cir.if %[[SUCC]] {
  // CIR:    cir.store align(16) %[[OUT]], %[[ODATA_PTR:.+]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  } else {
  // CIR:    %[[NULL:.+]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
  // CIR:    cir.store align(16) %[[NULL]], %[[ODATA_PTR]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  }
  // CIR:  %[[FLAG1:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct -> !u8i
  // CIR:  cir.store %[[FLAG1]], %[[RET_ADDR:.+]] : !u8i, !cir.ptr<!u8i>
  // CIR:  %[[RET:.+]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u8i>, !u8i
  // CIR:  cir.return %[[RET]] : !u8i

  // LLVM-LABEL: _mm_aesdec128kl_u8
  // LLVM:   %[[RESULT:.+]] = call { i8, <2 x i64> } @llvm.x86.aesdec128kl(<2 x i64> %[[IDATA:.+]], ptr %[[H:.+]])
  // LLVM:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // LLVM:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // LLVM:   %[[OUT:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 1
  // LLVM:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // LLVM: [[NO_ERROR]]:
  // LLVM:   store <2 x i64> %[[OUT]], ptr %[[ODATA_PTR:.+]], align 16
  // LLVM:   br label %[[EXIT:.+]]
  // LLVM: [[ERROR]]:
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // LLVM:   br label %[[EXIT]]
  // LLVM: [[EXIT]]:
  // LLVM:   %[[FLAG1:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // LLVM:   store i8 %[[FLAG1]], ptr %[[FLAG1_ADDR:.+]], align 1
  // LLVM:   %[[FLAG2:.+]] = load i8, ptr %[[FLAG1_ADDR]], align 1
  // LLVM:   store i8 %[[FLAG2:.+]], ptr %[[FLAG2_ADDR:.+]], align 1
  // LLVM:   %[[RET:.+]] = load i8, ptr %[[FLAG2_ADDR]], align 1
  // LLVM:   ret i8 %[[RET]]

  // OGCG-LABEL: test_mm_aesdec128kl_u8
  // OGCG:   %[[RESULT:.+]] = call { i8, <2 x i64> } @llvm.x86.aesdec128kl(<2 x i64> %[[IDATA:.+]], ptr %[[H:.+]])
  // OGCG:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // OGCG:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // OGCG:   %[[OUT:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 1
  // OGCG:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // OGCG: [[NO_ERROR]]:
  // OGCG:   store <2 x i64> %[[OUT]], ptr %[[ODATA_PTR:.+]], align 16
  // OGCG:   br label %[[EXIT:.+]]
  // OGCG: [[ERROR]]:
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // OGCG:   br label %[[EXIT]]
  // OGCG: [[EXIT]]:
  // OGCG:   %[[RET:.+]] = extractvalue { i8, <2 x i64> } %[[RESULT]], 0
  // OGCG:   ret i8 %[[RET]]
  return _mm_aesdec128kl_u8(odata, idata, h);
}

unsigned char test__mm_aesencwide256kl_u8(__m128i odata[8], const __m128i idata[8], const void* h) {
  // CIR-LABEL: _mm_aesencwide256kl_u8
  // CIR:  %[[ZERO:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:  %[[IN_ADDR0:.+]] = cir.ptr_stride %[[IDATA_ADDR:.+]], %[[ZERO]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA0:.+]] = cir.load align(16) %[[IN_ADDR0]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[ONE:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:  %[[IN_ADDR1:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[ONE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA1:.+]] = cir.load align(16) %[[IN_ADDR1]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[TWO:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:  %[[IN_ADDR2:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[TWO]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA2:.+]] = cir.load align(16) %[[IN_ADDR2]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[THREE:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:  %[[IN_ADDR3:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[THREE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA3:.+]] = cir.load align(16) %[[IN_ADDR3]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[FOUR:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:  %[[IN_ADDR4:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[FOUR]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA4:.+]] = cir.load align(16) %[[IN_ADDR4]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[FIVE:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:  %[[IN_ADDR5:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[FIVE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA5:.+]] = cir.load align(16) %[[IN_ADDR5]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[SIX:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:  %[[IN_ADDR6:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[SIX]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA6:.+]] = cir.load align(16) %[[IN_ADDR6]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[SEVEN:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:  %[[IN_ADDR7:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[SEVEN]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA7:.+]] = cir.load align(16) %[[IN_ADDR7]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[RESULT:.+]] = cir.call_llvm_intrinsic "x86.aesencwide256kl" %[[H_ADDR:.+]], %[[IN_DATA0]], %[[IN_DATA1]], %[[IN_DATA2]], %[[IN_DATA3]], %[[IN_DATA4]], %[[IN_DATA5]], %[[IN_DATA6]], %[[IN_DATA7]] : (!cir.ptr<!void>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>) -> !rec_anon_struct1
  // CIR:  %[[FLAG:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct1 -> !u8i
  // CIR:  %[[FLAG_BIT0:.+]] = cir.cast integral %[[FLAG]] : !u8i -> !cir.int<u, 1>
  // CIR:  %[[SUCC:.+]] = cir.cast int_to_bool %[[FLAG_BIT0]] : !cir.int<u, 1> -> !cir.bool
  // CIR:  cir.if %[[SUCC]] {
  // CIR:    %[[OUT_DATA0:.+]] = cir.extract_member %[[RESULT]][1] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[ZERO_1:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:    %[[OUT_ADDR0:.+]] = cir.ptr_stride %[[ODATA_ADDR:.+]], %[[ZERO_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA0]], %[[OUT_ADDR0]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA1:.+]] = cir.extract_member %[[RESULT]][2] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[ONE_1:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:    %[[OUT_ADDR1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ONE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA1]], %[[OUT_ADDR1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA2:.+]] = cir.extract_member %[[RESULT]][3] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[TWO_1:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:    %[[OUT_ADDR2:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[TWO_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA2]], %[[OUT_ADDR2]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA3:.+]] = cir.extract_member %[[RESULT]][4] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[THREE_1:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:    %[[OUT_ADDR3:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[THREE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA3]], %[[OUT_ADDR3]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA4:.+]] = cir.extract_member %[[RESULT]][5] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[FOUR_1:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:    %[[OUT_ADDR4:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FOUR_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA4]], %[[OUT_ADDR4]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA5:.+]] = cir.extract_member %[[RESULT]][6] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[FIVE_1:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:    %[[OUT_ADDR5:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FIVE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA5]], %[[OUT_ADDR5]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA6:.+]] = cir.extract_member %[[RESULT]][7] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[SIX_1:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:    %[[OUT_ADDR6:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SIX_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA6]], %[[OUT_ADDR6]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA7:.+]] = cir.extract_member %[[RESULT]][8] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[SEVEN_1:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:    %[[OUT_ADDR7:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SEVEN_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA7]], %[[OUT_ADDR7]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  } else {
  // CIR:    %[[NULL:.+]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
  // CIR:    %[[ZERO_2:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:    %[[OUT_ADDR0_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ZERO_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR0_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[ONE_2:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:    %[[OUT_ADDR1_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ONE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR1_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[TWO_2:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:    %[[OUT_ADDR2_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[TWO_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR2_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[THREE_2:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:    %[[OUT_ADDR3_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[THREE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR3_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[FOUR_2:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:    %[[OUT_ADDR4_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FOUR_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR4_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[FIVE_2:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:    %[[OUT_ADDR5_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FIVE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR5_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[SIX_2:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:    %[[OUT_ADDR6_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SIX_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR6_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[SEVEN_2:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:    %[[OUT_ADDR7_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SEVEN_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR7_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  }
  // CIR:  %[[FLAG1:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct1 -> !u8i
  // CIR:  cir.store %[[FLAG1]], %[[RET_ADDR:.+]] : !u8i, !cir.ptr<!u8i>
  // CIR:  %[[RET:.+]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u8i>, !u8i
  // CIR:  cir.return %[[RET]] : !u8i

  // LLVM-LABEL: _mm_aesencwide256kl_u8
  // LLVM:   %[[IN_DATA0:.+]] = load <2 x i64>, ptr %[[IDATA_ADDR:.+]], align 16
  // LLVM:   %[[IN_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 1
  // LLVM:   %[[IN_DATA1:.+]] = load <2 x i64>, ptr %[[IN_ADDR1]], align 16
  // LLVM:   %[[IN_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 2
  // LLVM:   %[[IN_DATA2:.+]] = load <2 x i64>, ptr %[[IN_ADDR2]], align 16
  // LLVM:   %[[IN_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 3
  // LLVM:   %[[IN_DATA3:.+]] = load <2 x i64>, ptr %[[IN_ADDR3]], align 16
  // LLVM:   %[[IN_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 4
  // LLVM:   %[[IN_DATA4:.+]] = load <2 x i64>, ptr %[[IN_ADDR4]], align 16
  // LLVM:   %[[IN_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 5
  // LLVM:   %[[IN_DATA5:.+]] = load <2 x i64>, ptr %[[IN_ADDR5]], align 16
  // LLVM:   %[[IN_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 6
  // LLVM:   %[[IN_DATA6:.+]] = load <2 x i64>, ptr %[[IN_ADDR6]], align 16
  // LLVM:   %[[IN_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 7
  // LLVM:   %[[IN_DATA7:.+]] = load <2 x i64>, ptr %[[IN_ADDR7]], align 16
  // LLVM:   %[[RESULT:.+]] = call { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.aesencwide256kl(ptr %[[H_ADDR:.+]], <2 x i64>  %[[IN_DATA0]], <2 x i64>  %[[IN_DATA1]], <2 x i64>  %[[IN_DATA2]], <2 x i64>  %[[IN_DATA3]], <2 x i64>  %[[IN_DATA4]], <2 x i64>  %[[IN_DATA5]], <2 x i64>  %[[IN_DATA6]], <2 x i64>  %[[IN_DATA7]])
  // LLVM:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // LLVM:   %[[SUCC:.+]] = trunc i8 %34 to i1
  // LLVM:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // LLVM: [[NO_ERROR]]:
  // LLVM:   %[[OUT_DATA0:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 1
  // LLVM:   store <2 x i64> %[[OUT_DATA0]], ptr %[[ODATA_PTR:.+]], align 16
  // LLVM:   %[[OUT_DATA1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 2
  // LLVM:   %[[OUT_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 1
  // LLVM:   store <2 x i64> %[[OUT_DATA1]], ptr %[[OUT_ADDR1]], align 16
  // LLVM:   %[[OUT_DATA2:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 3
  // LLVM:   %[[OUT_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 2
  // LLVM:   store <2 x i64> %[[OUT_DATA2]], ptr %[[OUT_ADDR2]], align 16
  // LLVM:   %[[OUT_DATA3:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 4
  // LLVM:   %[[OUT_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 3
  // LLVM:   store <2 x i64> %[[OUT_DATA3]], ptr %[[OUT_ADDR3]], align 16
  // LLVM:   %[[OUT_DATA4:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 5
  // LLVM:   %[[OUT_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 4
  // LLVM:   store <2 x i64> %[[OUT_DATA4]], ptr %[[OUT_ADDR4]], align 16
  // LLVM:   %[[OUT_DATA5:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 6
  // LLVM:   %[[OUT_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 5
  // LLVM:   store <2 x i64> %[[OUT_DATA5]], ptr %[[OUT_ADDR5]], align 16
  // LLVM:   %[[OUT_DATA6:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 7
  // LLVM:   %[[OUT_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 6
  // LLVM:   store <2 x i64> %[[OUT_DATA6]], ptr %[[OUT_ADDR6]], align 16
  // LLVM:   %[[OUT_DATA7:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 8
  // LLVM:   %[[OUT_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 7
  // LLVM:   store <2 x i64> %[[OUT_DATA7]], ptr %[[OUT_ADDR7]], align 16
  // LLVM:   br label %[[EXIT:.+]]
  // LLVM: [[ERROR]]:
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // LLVM:   %[[OUT_ADDR0_1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 1
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_1]], align 16
  // LLVM:   %[[OUT_ADDR0_2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 2
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_2]], align 16
  // LLVM:   %[[OUT_ADDR0_3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 3
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_3]], align 16
  // LLVM:   %[[OUT_ADDR0_4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 4
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_4]], align 16
  // LLVM:   %[[OUT_ADDR0_5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 5
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_5]], align 16
  // LLVM:   %[[OUT_ADDR0_6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 6
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_6]], align 16
  // LLVM:   %[[OUT_ADDR0_7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 7
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_7]], align 16
  // LLVM:   br label %[[EXIT]]
  // LLVM: [[EXIT]]:
  // LLVM:   %[[FLAG1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // LLVM:   store i8 %[[FLAG1]], ptr %[[FLAG1_ADDR:.+]], align 1
  // LLVM:   %[[FLAG2:.+]] = load i8, ptr %[[FLAG1_ADDR]], align 1
  // LLVM:   store i8 %[[FLAG2:.+]], ptr %[[FLAG2_ADDR:.+]], align 1
  // LLVM:   %[[RET:.+]] = load i8, ptr %[[FLAG2_ADDR]], align 1
  // LLVM:   ret i8 %[[RET]]

  // OGCG-LABEL: _mm_aesencwide256kl_u8
  // OGCG:   %[[IN_DATA0:.+]] = load <2 x i64>, ptr %[[IDATA_ADDR:.+]], align 16
  // OGCG:   %[[IN_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 1
  // OGCG:   %[[IN_DATA1:.+]] = load <2 x i64>, ptr %[[IN_ADDR1]], align 16
  // OGCG:   %[[IN_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 2
  // OGCG:   %[[IN_DATA2:.+]] = load <2 x i64>, ptr %[[IN_ADDR2]], align 16
  // OGCG:   %[[IN_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 3
  // OGCG:   %[[IN_DATA3:.+]] = load <2 x i64>, ptr %[[IN_ADDR3]], align 16
  // OGCG:   %[[IN_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 4
  // OGCG:   %[[IN_DATA4:.+]] = load <2 x i64>, ptr %[[IN_ADDR4]], align 16
  // OGCG:   %[[IN_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 5
  // OGCG:   %[[IN_DATA5:.+]] = load <2 x i64>, ptr %[[IN_ADDR5]], align 16
  // OGCG:   %[[IN_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 6
  // OGCG:   %[[IN_DATA6:.+]] = load <2 x i64>, ptr %[[IN_ADDR6]], align 16
  // OGCG:   %[[IN_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 7
  // OGCG:   %[[IN_DATA7:.+]] = load <2 x i64>, ptr %[[IN_ADDR7]], align 16
  // OGCG:   %[[RESULT:.+]] = call { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.aesencwide256kl(ptr %[[H_ADDR:.+]], <2 x i64>  %[[IN_DATA0]], <2 x i64>  %[[IN_DATA1]], <2 x i64>  %[[IN_DATA2]], <2 x i64>  %[[IN_DATA3]], <2 x i64>  %[[IN_DATA4]], <2 x i64>  %[[IN_DATA5]], <2 x i64>  %[[IN_DATA6]], <2 x i64>  %[[IN_DATA7]])
  // OGCG:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // OGCG:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // OGCG:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // OGCG: [[NO_ERROR]]:
  // OGCG:   %[[OUT_DATA0:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 1
  // OGCG:   store <2 x i64> %[[OUT_DATA0]], ptr %[[ODATA_PTR:.+]], align 16
  // OGCG:   %[[OUT_DATA1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 2
  // OGCG:   %[[OUT_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 1
  // OGCG:   store <2 x i64> %[[OUT_DATA1]], ptr %[[OUT_ADDR1]], align 16
  // OGCG:   %[[OUT_DATA2:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 3
  // OGCG:   %[[OUT_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 2
  // OGCG:   store <2 x i64> %[[OUT_DATA2]], ptr %[[OUT_ADDR2]], align 16
  // OGCG:   %[[OUT_DATA3:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 4
  // OGCG:   %[[OUT_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 3
  // OGCG:   store <2 x i64> %[[OUT_DATA3]], ptr %[[OUT_ADDR3]], align 16
  // OGCG:   %[[OUT_DATA4:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 5
  // OGCG:   %[[OUT_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 4
  // OGCG:   store <2 x i64> %[[OUT_DATA4]], ptr %[[OUT_ADDR4]], align 16
  // OGCG:   %[[OUT_DATA5:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 6
  // OGCG:   %[[OUT_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 5
  // OGCG:   store <2 x i64> %[[OUT_DATA5]], ptr %[[OUT_ADDR5]], align 16
  // OGCG:   %[[OUT_DATA6:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 7
  // OGCG:   %[[OUT_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 6
  // OGCG:   store <2 x i64> %[[OUT_DATA6]], ptr %[[OUT_ADDR6]], align 16
  // OGCG:   %[[OUT_DATA7:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 8
  // OGCG:   %[[OUT_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 7
  // OGCG:   store <2 x i64> %[[OUT_DATA7]], ptr %[[OUT_ADDR7]], align 16
  // OGCG:   br label %[[EXIT:.+]]
  // OGCG: [[ERROR]]:
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // OGCG:   %[[OUT_ADDR0_1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 1
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_1]], align 16
  // OGCG:   %[[OUT_ADDR0_2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 2
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_2]], align 16
  // OGCG:   %[[OUT_ADDR0_3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 3
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_3]], align 16
  // OGCG:   %[[OUT_ADDR0_4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 4
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_4]], align 16
  // OGCG:   %[[OUT_ADDR0_5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 5
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_5]], align 16
  // OGCG:   %[[OUT_ADDR0_6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 6
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_6]], align 16
  // OGCG:   %[[OUT_ADDR0_7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 7
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_7]], align 16
  // OGCG:   br label %[[EXIT]]
  // OGCG: [[EXIT]]:
  // OGCG:   %[[RET:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // OGCG:   ret i8 %[[RET]]
  return _mm_aesencwide256kl_u8(odata, idata, h);
}

unsigned char test__mm_aesdecwide256kl_u8(__m128i odata[8], const __m128i idata[8], const void* h) {
  // CIR-LABEL: _mm_aesdecwide256kl_u8
  // CIR:  %[[ZERO:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:  %[[IN_ADDR0:.+]] = cir.ptr_stride %[[IDATA_ADDR:.+]], %[[ZERO]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA0:.+]] = cir.load align(16) %[[IN_ADDR0]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[ONE:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:  %[[IN_ADDR1:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[ONE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA1:.+]] = cir.load align(16) %[[IN_ADDR1]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[TWO:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:  %[[IN_ADDR2:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[TWO]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA2:.+]] = cir.load align(16) %[[IN_ADDR2]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[THREE:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:  %[[IN_ADDR3:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[THREE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA3:.+]] = cir.load align(16) %[[IN_ADDR3]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[FOUR:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:  %[[IN_ADDR4:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[FOUR]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA4:.+]] = cir.load align(16) %[[IN_ADDR4]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[FIVE:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:  %[[IN_ADDR5:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[FIVE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA5:.+]] = cir.load align(16) %[[IN_ADDR5]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[SIX:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:  %[[IN_ADDR6:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[SIX]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA6:.+]] = cir.load align(16) %[[IN_ADDR6]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[SEVEN:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:  %[[IN_ADDR7:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[SEVEN]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA7:.+]] = cir.load align(16) %[[IN_ADDR7]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[RESULT:.+]] = cir.call_llvm_intrinsic "x86.aesdecwide256kl" %[[H_ADDR:.+]], %[[IN_DATA0]], %[[IN_DATA1]], %[[IN_DATA2]], %[[IN_DATA3]], %[[IN_DATA4]], %[[IN_DATA5]], %[[IN_DATA6]], %[[IN_DATA7]] : (!cir.ptr<!void>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>) -> !rec_anon_struct1
  // CIR:  %[[FLAG:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct1 -> !u8i
  // CIR:  %[[FLAG_BIT0:.+]] = cir.cast integral %[[FLAG]] : !u8i -> !cir.int<u, 1>
  // CIR:  %[[SUCC:.+]] = cir.cast int_to_bool %[[FLAG_BIT0]] : !cir.int<u, 1> -> !cir.bool
  // CIR:  cir.if %[[SUCC]] {
  // CIR:    %[[OUT_DATA0:.+]] = cir.extract_member %[[RESULT]][1] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[ZERO_1:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:    %[[OUT_ADDR0:.+]] = cir.ptr_stride %[[ODATA_ADDR:.+]], %[[ZERO_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA0]], %[[OUT_ADDR0]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA1:.+]] = cir.extract_member %[[RESULT]][2] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[ONE_1:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:    %[[OUT_ADDR1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ONE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA1]], %[[OUT_ADDR1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA2:.+]] = cir.extract_member %[[RESULT]][3] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[TWO_1:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:    %[[OUT_ADDR2:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[TWO_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA2]], %[[OUT_ADDR2]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA3:.+]] = cir.extract_member %[[RESULT]][4] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[THREE_1:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:    %[[OUT_ADDR3:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[THREE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA3]], %[[OUT_ADDR3]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA4:.+]] = cir.extract_member %[[RESULT]][5] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[FOUR_1:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:    %[[OUT_ADDR4:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FOUR_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA4]], %[[OUT_ADDR4]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA5:.+]] = cir.extract_member %[[RESULT]][6] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[FIVE_1:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:    %[[OUT_ADDR5:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FIVE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA5]], %[[OUT_ADDR5]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA6:.+]] = cir.extract_member %[[RESULT]][7] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[SIX_1:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:    %[[OUT_ADDR6:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SIX_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA6]], %[[OUT_ADDR6]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA7:.+]] = cir.extract_member %[[RESULT]][8] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[SEVEN_1:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:    %[[OUT_ADDR7:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SEVEN_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA7]], %[[OUT_ADDR7]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  } else {
  // CIR:    %[[NULL:.+]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
  // CIR:    %[[ZERO_2:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:    %[[OUT_ADDR0_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ZERO_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR0_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[ONE_2:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:    %[[OUT_ADDR1_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ONE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR1_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[TWO_2:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:    %[[OUT_ADDR2_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[TWO_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR2_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[THREE_2:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:    %[[OUT_ADDR3_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[THREE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR3_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[FOUR_2:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:    %[[OUT_ADDR4_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FOUR_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR4_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[FIVE_2:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:    %[[OUT_ADDR5_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FIVE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR5_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[SIX_2:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:    %[[OUT_ADDR6_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SIX_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR6_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[SEVEN_2:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:    %[[OUT_ADDR7_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SEVEN_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR7_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  }
  // CIR:  %[[FLAG1:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct1 -> !u8i
  // CIR:  cir.store %[[FLAG1]], %[[RET_ADDR:.+]] : !u8i, !cir.ptr<!u8i>
  // CIR:  %[[RET:.+]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u8i>, !u8i
  // CIR:  cir.return %[[RET]] : !u8i

  // LLVM-LABEL: _mm_aesdecwide256kl_u8
  // LLVM:   %[[IN_DATA0:.+]] = load <2 x i64>, ptr %[[IDATA_ADDR:.+]], align 16
  // LLVM:   %[[IN_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 1
  // LLVM:   %[[IN_DATA1:.+]] = load <2 x i64>, ptr %[[IN_ADDR1]], align 16
  // LLVM:   %[[IN_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 2
  // LLVM:   %[[IN_DATA2:.+]] = load <2 x i64>, ptr %[[IN_ADDR2]], align 16
  // LLVM:   %[[IN_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 3
  // LLVM:   %[[IN_DATA3:.+]] = load <2 x i64>, ptr %[[IN_ADDR3]], align 16
  // LLVM:   %[[IN_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 4
  // LLVM:   %[[IN_DATA4:.+]] = load <2 x i64>, ptr %[[IN_ADDR4]], align 16
  // LLVM:   %[[IN_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 5
  // LLVM:   %[[IN_DATA5:.+]] = load <2 x i64>, ptr %[[IN_ADDR5]], align 16
  // LLVM:   %[[IN_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 6
  // LLVM:   %[[IN_DATA6:.+]] = load <2 x i64>, ptr %[[IN_ADDR6]], align 16
  // LLVM:   %[[IN_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 7
  // LLVM:   %[[IN_DATA7:.+]] = load <2 x i64>, ptr %[[IN_ADDR7]], align 16
  // LLVM:   %[[RESULT:.+]] = call { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.aesdecwide256kl(ptr %[[H_ADDR:.+]], <2 x i64>  %[[IN_DATA0]], <2 x i64>  %[[IN_DATA1]], <2 x i64>  %[[IN_DATA2]], <2 x i64>  %[[IN_DATA3]], <2 x i64>  %[[IN_DATA4]], <2 x i64>  %[[IN_DATA5]], <2 x i64>  %[[IN_DATA6]], <2 x i64>  %[[IN_DATA7]])
  // LLVM:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // LLVM:   %[[SUCC:.+]] = trunc i8 %34 to i1
  // LLVM:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // LLVM: [[NO_ERROR]]:
  // LLVM:   %[[OUT_DATA0:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 1
  // LLVM:   store <2 x i64> %[[OUT_DATA0]], ptr %[[ODATA_PTR:.+]], align 16
  // LLVM:   %[[OUT_DATA1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 2
  // LLVM:   %[[OUT_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 1
  // LLVM:   store <2 x i64> %[[OUT_DATA1]], ptr %[[OUT_ADDR1]], align 16
  // LLVM:   %[[OUT_DATA2:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 3
  // LLVM:   %[[OUT_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 2
  // LLVM:   store <2 x i64> %[[OUT_DATA2]], ptr %[[OUT_ADDR2]], align 16
  // LLVM:   %[[OUT_DATA3:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 4
  // LLVM:   %[[OUT_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 3
  // LLVM:   store <2 x i64> %[[OUT_DATA3]], ptr %[[OUT_ADDR3]], align 16
  // LLVM:   %[[OUT_DATA4:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 5
  // LLVM:   %[[OUT_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 4
  // LLVM:   store <2 x i64> %[[OUT_DATA4]], ptr %[[OUT_ADDR4]], align 16
  // LLVM:   %[[OUT_DATA5:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 6
  // LLVM:   %[[OUT_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 5
  // LLVM:   store <2 x i64> %[[OUT_DATA5]], ptr %[[OUT_ADDR5]], align 16
  // LLVM:   %[[OUT_DATA6:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 7
  // LLVM:   %[[OUT_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 6
  // LLVM:   store <2 x i64> %[[OUT_DATA6]], ptr %[[OUT_ADDR6]], align 16
  // LLVM:   %[[OUT_DATA7:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 8
  // LLVM:   %[[OUT_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 7
  // LLVM:   store <2 x i64> %[[OUT_DATA7]], ptr %[[OUT_ADDR7]], align 16
  // LLVM:   br label %[[EXIT:.+]]
  // LLVM: [[ERROR]]:
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // LLVM:   %[[OUT_ADDR0_1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 1
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_1]], align 16
  // LLVM:   %[[OUT_ADDR0_2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 2
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_2]], align 16
  // LLVM:   %[[OUT_ADDR0_3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 3
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_3]], align 16
  // LLVM:   %[[OUT_ADDR0_4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 4
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_4]], align 16
  // LLVM:   %[[OUT_ADDR0_5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 5
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_5]], align 16
  // LLVM:   %[[OUT_ADDR0_6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 6
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_6]], align 16
  // LLVM:   %[[OUT_ADDR0_7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 7
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_7]], align 16
  // LLVM:   br label %[[EXIT]]
  // LLVM: [[EXIT]]:
  // LLVM:   %[[FLAG1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // LLVM:   store i8 %[[FLAG1]], ptr %[[FLAG1_ADDR:.+]], align 1
  // LLVM:   %[[FLAG2:.+]] = load i8, ptr %[[FLAG1_ADDR]], align 1
  // LLVM:   store i8 %[[FLAG2:.+]], ptr %[[FLAG2_ADDR:.+]], align 1
  // LLVM:   %[[RET:.+]] = load i8, ptr %[[FLAG2_ADDR]], align 1
  // LLVM:   ret i8 %[[RET]]

  // OGCG-LABEL: _mm_aesdecwide256kl_u8
  // OGCG:   %[[IN_DATA0:.+]] = load <2 x i64>, ptr %[[IDATA_ADDR:.+]], align 16
  // OGCG:   %[[IN_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 1
  // OGCG:   %[[IN_DATA1:.+]] = load <2 x i64>, ptr %[[IN_ADDR1]], align 16
  // OGCG:   %[[IN_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 2
  // OGCG:   %[[IN_DATA2:.+]] = load <2 x i64>, ptr %[[IN_ADDR2]], align 16
  // OGCG:   %[[IN_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 3
  // OGCG:   %[[IN_DATA3:.+]] = load <2 x i64>, ptr %[[IN_ADDR3]], align 16
  // OGCG:   %[[IN_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 4
  // OGCG:   %[[IN_DATA4:.+]] = load <2 x i64>, ptr %[[IN_ADDR4]], align 16
  // OGCG:   %[[IN_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 5
  // OGCG:   %[[IN_DATA5:.+]] = load <2 x i64>, ptr %[[IN_ADDR5]], align 16
  // OGCG:   %[[IN_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 6
  // OGCG:   %[[IN_DATA6:.+]] = load <2 x i64>, ptr %[[IN_ADDR6]], align 16
  // OGCG:   %[[IN_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 7
  // OGCG:   %[[IN_DATA7:.+]] = load <2 x i64>, ptr %[[IN_ADDR7]], align 16
  // OGCG:   %[[RESULT:.+]] = call { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.aesdecwide256kl(ptr %[[H_ADDR:.+]], <2 x i64>  %[[IN_DATA0]], <2 x i64>  %[[IN_DATA1]], <2 x i64>  %[[IN_DATA2]], <2 x i64>  %[[IN_DATA3]], <2 x i64>  %[[IN_DATA4]], <2 x i64>  %[[IN_DATA5]], <2 x i64>  %[[IN_DATA6]], <2 x i64>  %[[IN_DATA7]])
  // OGCG:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // OGCG:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // OGCG:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // OGCG: [[NO_ERROR]]:
  // OGCG:   %[[OUT_DATA0:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 1
  // OGCG:   store <2 x i64> %[[OUT_DATA0]], ptr %[[ODATA_PTR:.+]], align 16
  // OGCG:   %[[OUT_DATA1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 2
  // OGCG:   %[[OUT_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 1
  // OGCG:   store <2 x i64> %[[OUT_DATA1]], ptr %[[OUT_ADDR1]], align 16
  // OGCG:   %[[OUT_DATA2:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 3
  // OGCG:   %[[OUT_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 2
  // OGCG:   store <2 x i64> %[[OUT_DATA2]], ptr %[[OUT_ADDR2]], align 16
  // OGCG:   %[[OUT_DATA3:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 4
  // OGCG:   %[[OUT_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 3
  // OGCG:   store <2 x i64> %[[OUT_DATA3]], ptr %[[OUT_ADDR3]], align 16
  // OGCG:   %[[OUT_DATA4:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 5
  // OGCG:   %[[OUT_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 4
  // OGCG:   store <2 x i64> %[[OUT_DATA4]], ptr %[[OUT_ADDR4]], align 16
  // OGCG:   %[[OUT_DATA5:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 6
  // OGCG:   %[[OUT_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 5
  // OGCG:   store <2 x i64> %[[OUT_DATA5]], ptr %[[OUT_ADDR5]], align 16
  // OGCG:   %[[OUT_DATA6:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 7
  // OGCG:   %[[OUT_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 6
  // OGCG:   store <2 x i64> %[[OUT_DATA6]], ptr %[[OUT_ADDR6]], align 16
  // OGCG:   %[[OUT_DATA7:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 8
  // OGCG:   %[[OUT_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 7
  // OGCG:   store <2 x i64> %[[OUT_DATA7]], ptr %[[OUT_ADDR7]], align 16
  // OGCG:   br label %[[EXIT:.+]]
  // OGCG: [[ERROR]]:
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // OGCG:   %[[OUT_ADDR0_1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 1
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_1]], align 16
  // OGCG:   %[[OUT_ADDR0_2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 2
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_2]], align 16
  // OGCG:   %[[OUT_ADDR0_3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 3
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_3]], align 16
  // OGCG:   %[[OUT_ADDR0_4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 4
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_4]], align 16
  // OGCG:   %[[OUT_ADDR0_5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 5
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_5]], align 16
  // OGCG:   %[[OUT_ADDR0_6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 6
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_6]], align 16
  // OGCG:   %[[OUT_ADDR0_7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 7
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_7]], align 16
  // OGCG:   br label %[[EXIT]]
  // OGCG: [[EXIT]]:
  // OGCG:   %[[RET:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // OGCG:   ret i8 %[[RET]]
  return _mm_aesdecwide256kl_u8(odata, idata, h);
}

unsigned char test__mm_aesencwide128kl_u8(__m128i odata[8], const __m128i idata[8], const void* h) {
  // CIR-LABEL: _mm_aesencwide128kl_u8
  // CIR:  %[[ZERO:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:  %[[IN_ADDR0:.+]] = cir.ptr_stride %[[IDATA_ADDR:.+]], %[[ZERO]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA0:.+]] = cir.load align(16) %[[IN_ADDR0]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[ONE:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:  %[[IN_ADDR1:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[ONE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA1:.+]] = cir.load align(16) %[[IN_ADDR1]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[TWO:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:  %[[IN_ADDR2:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[TWO]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA2:.+]] = cir.load align(16) %[[IN_ADDR2]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[THREE:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:  %[[IN_ADDR3:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[THREE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA3:.+]] = cir.load align(16) %[[IN_ADDR3]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[FOUR:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:  %[[IN_ADDR4:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[FOUR]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA4:.+]] = cir.load align(16) %[[IN_ADDR4]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[FIVE:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:  %[[IN_ADDR5:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[FIVE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA5:.+]] = cir.load align(16) %[[IN_ADDR5]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[SIX:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:  %[[IN_ADDR6:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[SIX]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA6:.+]] = cir.load align(16) %[[IN_ADDR6]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[SEVEN:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:  %[[IN_ADDR7:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[SEVEN]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA7:.+]] = cir.load align(16) %[[IN_ADDR7]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[RESULT:.+]] = cir.call_llvm_intrinsic "x86.aesencwide128kl" %[[H_ADDR:.+]], %[[IN_DATA0]], %[[IN_DATA1]], %[[IN_DATA2]], %[[IN_DATA3]], %[[IN_DATA4]], %[[IN_DATA5]], %[[IN_DATA6]], %[[IN_DATA7]] : (!cir.ptr<!void>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>) -> !rec_anon_struct1
  // CIR:  %[[FLAG:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct1 -> !u8i
  // CIR:  %[[FLAG_BIT0:.+]] = cir.cast integral %[[FLAG]] : !u8i -> !cir.int<u, 1>
  // CIR:  %[[SUCC:.+]] = cir.cast int_to_bool %[[FLAG_BIT0]] : !cir.int<u, 1> -> !cir.bool
  // CIR:  cir.if %[[SUCC]] {
  // CIR:    %[[OUT_DATA0:.+]] = cir.extract_member %[[RESULT]][1] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[ZERO_1:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:    %[[OUT_ADDR0:.+]] = cir.ptr_stride %[[ODATA_ADDR:.+]], %[[ZERO_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA0]], %[[OUT_ADDR0]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA1:.+]] = cir.extract_member %[[RESULT]][2] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[ONE_1:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:    %[[OUT_ADDR1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ONE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA1]], %[[OUT_ADDR1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA2:.+]] = cir.extract_member %[[RESULT]][3] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[TWO_1:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:    %[[OUT_ADDR2:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[TWO_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA2]], %[[OUT_ADDR2]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA3:.+]] = cir.extract_member %[[RESULT]][4] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[THREE_1:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:    %[[OUT_ADDR3:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[THREE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA3]], %[[OUT_ADDR3]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA4:.+]] = cir.extract_member %[[RESULT]][5] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[FOUR_1:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:    %[[OUT_ADDR4:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FOUR_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA4]], %[[OUT_ADDR4]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA5:.+]] = cir.extract_member %[[RESULT]][6] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[FIVE_1:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:    %[[OUT_ADDR5:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FIVE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA5]], %[[OUT_ADDR5]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA6:.+]] = cir.extract_member %[[RESULT]][7] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[SIX_1:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:    %[[OUT_ADDR6:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SIX_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA6]], %[[OUT_ADDR6]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA7:.+]] = cir.extract_member %[[RESULT]][8] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[SEVEN_1:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:    %[[OUT_ADDR7:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SEVEN_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA7]], %[[OUT_ADDR7]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  } else {
  // CIR:    %[[NULL:.+]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
  // CIR:    %[[ZERO_2:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:    %[[OUT_ADDR0_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ZERO_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR0_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[ONE_2:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:    %[[OUT_ADDR1_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ONE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR1_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[TWO_2:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:    %[[OUT_ADDR2_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[TWO_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR2_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[THREE_2:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:    %[[OUT_ADDR3_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[THREE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR3_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[FOUR_2:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:    %[[OUT_ADDR4_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FOUR_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR4_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[FIVE_2:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:    %[[OUT_ADDR5_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FIVE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR5_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[SIX_2:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:    %[[OUT_ADDR6_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SIX_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR6_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[SEVEN_2:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:    %[[OUT_ADDR7_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SEVEN_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR7_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  }
  // CIR:  %[[FLAG1:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct1 -> !u8i
  // CIR:  cir.store %[[FLAG1]], %[[RET_ADDR:.+]] : !u8i, !cir.ptr<!u8i>
  // CIR:  %[[RET:.+]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u8i>, !u8i
  // CIR:  cir.return %[[RET]] : !u8i

  // LLVM-LABEL: _mm_aesencwide128kl_u8
  // LLVM:   %[[IN_DATA0:.+]] = load <2 x i64>, ptr %[[IDATA_ADDR:.+]], align 16
  // LLVM:   %[[IN_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 1
  // LLVM:   %[[IN_DATA1:.+]] = load <2 x i64>, ptr %[[IN_ADDR1]], align 16
  // LLVM:   %[[IN_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 2
  // LLVM:   %[[IN_DATA2:.+]] = load <2 x i64>, ptr %[[IN_ADDR2]], align 16
  // LLVM:   %[[IN_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 3
  // LLVM:   %[[IN_DATA3:.+]] = load <2 x i64>, ptr %[[IN_ADDR3]], align 16
  // LLVM:   %[[IN_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 4
  // LLVM:   %[[IN_DATA4:.+]] = load <2 x i64>, ptr %[[IN_ADDR4]], align 16
  // LLVM:   %[[IN_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 5
  // LLVM:   %[[IN_DATA5:.+]] = load <2 x i64>, ptr %[[IN_ADDR5]], align 16
  // LLVM:   %[[IN_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 6
  // LLVM:   %[[IN_DATA6:.+]] = load <2 x i64>, ptr %[[IN_ADDR6]], align 16
  // LLVM:   %[[IN_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 7
  // LLVM:   %[[IN_DATA7:.+]] = load <2 x i64>, ptr %[[IN_ADDR7]], align 16
  // LLVM:   %[[RESULT:.+]] = call { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.aesencwide128kl(ptr %[[H_ADDR:.+]], <2 x i64>  %[[IN_DATA0]], <2 x i64>  %[[IN_DATA1]], <2 x i64>  %[[IN_DATA2]], <2 x i64>  %[[IN_DATA3]], <2 x i64>  %[[IN_DATA4]], <2 x i64>  %[[IN_DATA5]], <2 x i64>  %[[IN_DATA6]], <2 x i64>  %[[IN_DATA7]])
  // LLVM:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // LLVM:   %[[SUCC:.+]] = trunc i8 %34 to i1
  // LLVM:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // LLVM: [[NO_ERROR]]:
  // LLVM:   %[[OUT_DATA0:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 1
  // LLVM:   store <2 x i64> %[[OUT_DATA0]], ptr %[[ODATA_PTR:.+]], align 16
  // LLVM:   %[[OUT_DATA1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 2
  // LLVM:   %[[OUT_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 1
  // LLVM:   store <2 x i64> %[[OUT_DATA1]], ptr %[[OUT_ADDR1]], align 16
  // LLVM:   %[[OUT_DATA2:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 3
  // LLVM:   %[[OUT_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 2
  // LLVM:   store <2 x i64> %[[OUT_DATA2]], ptr %[[OUT_ADDR2]], align 16
  // LLVM:   %[[OUT_DATA3:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 4
  // LLVM:   %[[OUT_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 3
  // LLVM:   store <2 x i64> %[[OUT_DATA3]], ptr %[[OUT_ADDR3]], align 16
  // LLVM:   %[[OUT_DATA4:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 5
  // LLVM:   %[[OUT_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 4
  // LLVM:   store <2 x i64> %[[OUT_DATA4]], ptr %[[OUT_ADDR4]], align 16
  // LLVM:   %[[OUT_DATA5:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 6
  // LLVM:   %[[OUT_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 5
  // LLVM:   store <2 x i64> %[[OUT_DATA5]], ptr %[[OUT_ADDR5]], align 16
  // LLVM:   %[[OUT_DATA6:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 7
  // LLVM:   %[[OUT_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 6
  // LLVM:   store <2 x i64> %[[OUT_DATA6]], ptr %[[OUT_ADDR6]], align 16
  // LLVM:   %[[OUT_DATA7:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 8
  // LLVM:   %[[OUT_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 7
  // LLVM:   store <2 x i64> %[[OUT_DATA7]], ptr %[[OUT_ADDR7]], align 16
  // LLVM:   br label %[[EXIT:.+]]
  // LLVM: [[ERROR]]:
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // LLVM:   %[[OUT_ADDR0_1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 1
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_1]], align 16
  // LLVM:   %[[OUT_ADDR0_2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 2
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_2]], align 16
  // LLVM:   %[[OUT_ADDR0_3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 3
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_3]], align 16
  // LLVM:   %[[OUT_ADDR0_4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 4
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_4]], align 16
  // LLVM:   %[[OUT_ADDR0_5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 5
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_5]], align 16
  // LLVM:   %[[OUT_ADDR0_6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 6
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_6]], align 16
  // LLVM:   %[[OUT_ADDR0_7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 7
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_7]], align 16
  // LLVM:   br label %[[EXIT]]
  // LLVM: [[EXIT]]:
  // LLVM:   %[[FLAG1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // LLVM:   store i8 %[[FLAG1]], ptr %[[FLAG1_ADDR:.+]], align 1
  // LLVM:   %[[FLAG2:.+]] = load i8, ptr %[[FLAG1_ADDR]], align 1
  // LLVM:   store i8 %[[FLAG2:.+]], ptr %[[FLAG2_ADDR:.+]], align 1
  // LLVM:   %[[RET:.+]] = load i8, ptr %[[FLAG2_ADDR]], align 1
  // LLVM:   ret i8 %[[RET]]

  // OGCG-LABEL: _mm_aesencwide128kl_u8
  // OGCG:   %[[IN_DATA0:.+]] = load <2 x i64>, ptr %[[IDATA_ADDR:.+]], align 16
  // OGCG:   %[[IN_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 1
  // OGCG:   %[[IN_DATA1:.+]] = load <2 x i64>, ptr %[[IN_ADDR1]], align 16
  // OGCG:   %[[IN_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 2
  // OGCG:   %[[IN_DATA2:.+]] = load <2 x i64>, ptr %[[IN_ADDR2]], align 16
  // OGCG:   %[[IN_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 3
  // OGCG:   %[[IN_DATA3:.+]] = load <2 x i64>, ptr %[[IN_ADDR3]], align 16
  // OGCG:   %[[IN_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 4
  // OGCG:   %[[IN_DATA4:.+]] = load <2 x i64>, ptr %[[IN_ADDR4]], align 16
  // OGCG:   %[[IN_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 5
  // OGCG:   %[[IN_DATA5:.+]] = load <2 x i64>, ptr %[[IN_ADDR5]], align 16
  // OGCG:   %[[IN_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 6
  // OGCG:   %[[IN_DATA6:.+]] = load <2 x i64>, ptr %[[IN_ADDR6]], align 16
  // OGCG:   %[[IN_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 7
  // OGCG:   %[[IN_DATA7:.+]] = load <2 x i64>, ptr %[[IN_ADDR7]], align 16
  // OGCG:   %[[RESULT:.+]] = call { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.aesencwide128kl(ptr %[[H_ADDR:.+]], <2 x i64>  %[[IN_DATA0]], <2 x i64>  %[[IN_DATA1]], <2 x i64>  %[[IN_DATA2]], <2 x i64>  %[[IN_DATA3]], <2 x i64>  %[[IN_DATA4]], <2 x i64>  %[[IN_DATA5]], <2 x i64>  %[[IN_DATA6]], <2 x i64>  %[[IN_DATA7]])
  // OGCG:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // OGCG:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // OGCG:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // OGCG: [[NO_ERROR]]:
  // OGCG:   %[[OUT_DATA0:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 1
  // OGCG:   store <2 x i64> %[[OUT_DATA0]], ptr %[[ODATA_PTR:.+]], align 16
  // OGCG:   %[[OUT_DATA1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 2
  // OGCG:   %[[OUT_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 1
  // OGCG:   store <2 x i64> %[[OUT_DATA1]], ptr %[[OUT_ADDR1]], align 16
  // OGCG:   %[[OUT_DATA2:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 3
  // OGCG:   %[[OUT_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 2
  // OGCG:   store <2 x i64> %[[OUT_DATA2]], ptr %[[OUT_ADDR2]], align 16
  // OGCG:   %[[OUT_DATA3:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 4
  // OGCG:   %[[OUT_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 3
  // OGCG:   store <2 x i64> %[[OUT_DATA3]], ptr %[[OUT_ADDR3]], align 16
  // OGCG:   %[[OUT_DATA4:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 5
  // OGCG:   %[[OUT_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 4
  // OGCG:   store <2 x i64> %[[OUT_DATA4]], ptr %[[OUT_ADDR4]], align 16
  // OGCG:   %[[OUT_DATA5:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 6
  // OGCG:   %[[OUT_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 5
  // OGCG:   store <2 x i64> %[[OUT_DATA5]], ptr %[[OUT_ADDR5]], align 16
  // OGCG:   %[[OUT_DATA6:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 7
  // OGCG:   %[[OUT_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 6
  // OGCG:   store <2 x i64> %[[OUT_DATA6]], ptr %[[OUT_ADDR6]], align 16
  // OGCG:   %[[OUT_DATA7:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 8
  // OGCG:   %[[OUT_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 7
  // OGCG:   store <2 x i64> %[[OUT_DATA7]], ptr %[[OUT_ADDR7]], align 16
  // OGCG:   br label %[[EXIT:.+]]
  // OGCG: [[ERROR]]:
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // OGCG:   %[[OUT_ADDR0_1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 1
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_1]], align 16
  // OGCG:   %[[OUT_ADDR0_2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 2
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_2]], align 16
  // OGCG:   %[[OUT_ADDR0_3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 3
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_3]], align 16
  // OGCG:   %[[OUT_ADDR0_4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 4
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_4]], align 16
  // OGCG:   %[[OUT_ADDR0_5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 5
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_5]], align 16
  // OGCG:   %[[OUT_ADDR0_6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 6
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_6]], align 16
  // OGCG:   %[[OUT_ADDR0_7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 7
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_7]], align 16
  // OGCG:   br label %[[EXIT]]
  // OGCG: [[EXIT]]:
  // OGCG:   %[[RET:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // OGCG:   ret i8 %[[RET]]
  return _mm_aesencwide128kl_u8(odata, idata, h);
}

unsigned char test__mm_aesdecwide128kl_u8(__m128i odata[8], const __m128i idata[8], const void* h) {
  // CIR-LABEL: _mm_aesdecwide128kl_u8
  // CIR:  %[[ZERO:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:  %[[IN_ADDR0:.+]] = cir.ptr_stride %[[IDATA_ADDR:.+]], %[[ZERO]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA0:.+]] = cir.load align(16) %[[IN_ADDR0]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[ONE:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:  %[[IN_ADDR1:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[ONE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA1:.+]] = cir.load align(16) %[[IN_ADDR1]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[TWO:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:  %[[IN_ADDR2:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[TWO]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA2:.+]] = cir.load align(16) %[[IN_ADDR2]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[THREE:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:  %[[IN_ADDR3:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[THREE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA3:.+]] = cir.load align(16) %[[IN_ADDR3]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[FOUR:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:  %[[IN_ADDR4:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[FOUR]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA4:.+]] = cir.load align(16) %[[IN_ADDR4]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[FIVE:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:  %[[IN_ADDR5:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[FIVE]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA5:.+]] = cir.load align(16) %[[IN_ADDR5]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[SIX:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:  %[[IN_ADDR6:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[SIX]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA6:.+]] = cir.load align(16) %[[IN_ADDR6]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[SEVEN:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:  %[[IN_ADDR7:.+]] = cir.ptr_stride %[[IDATA_ADDR]], %[[SEVEN]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  %[[IN_DATA7:.+]] = cir.load align(16) %[[IN_ADDR7]] : !cir.ptr<!cir.vector<2 x !s64i>>, !cir.vector<2 x !s64i>
  // CIR:  %[[RESULT:.+]] = cir.call_llvm_intrinsic "x86.aesdecwide128kl" %[[H_ADDR:.+]], %[[IN_DATA0]], %[[IN_DATA1]], %[[IN_DATA2]], %[[IN_DATA3]], %[[IN_DATA4]], %[[IN_DATA5]], %[[IN_DATA6]], %[[IN_DATA7]] : (!cir.ptr<!void>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>, !cir.vector<2 x !s64i>) -> !rec_anon_struct1
  // CIR:  %[[FLAG:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct1 -> !u8i
  // CIR:  %[[FLAG_BIT0:.+]] = cir.cast integral %[[FLAG]] : !u8i -> !cir.int<u, 1>
  // CIR:  %[[SUCC:.+]] = cir.cast int_to_bool %[[FLAG_BIT0]] : !cir.int<u, 1> -> !cir.bool
  // CIR:  cir.if %[[SUCC]] {
  // CIR:    %[[OUT_DATA0:.+]] = cir.extract_member %[[RESULT]][1] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[ZERO_1:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:    %[[OUT_ADDR0:.+]] = cir.ptr_stride %[[ODATA_ADDR:.+]], %[[ZERO_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA0]], %[[OUT_ADDR0]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA1:.+]] = cir.extract_member %[[RESULT]][2] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[ONE_1:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:    %[[OUT_ADDR1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ONE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA1]], %[[OUT_ADDR1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA2:.+]] = cir.extract_member %[[RESULT]][3] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[TWO_1:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:    %[[OUT_ADDR2:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[TWO_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA2]], %[[OUT_ADDR2]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA3:.+]] = cir.extract_member %[[RESULT]][4] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[THREE_1:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:    %[[OUT_ADDR3:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[THREE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA3]], %[[OUT_ADDR3]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA4:.+]] = cir.extract_member %[[RESULT]][5] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[FOUR_1:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:    %[[OUT_ADDR4:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FOUR_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA4]], %[[OUT_ADDR4]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA5:.+]] = cir.extract_member %[[RESULT]][6] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[FIVE_1:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:    %[[OUT_ADDR5:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FIVE_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA5]], %[[OUT_ADDR5]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA6:.+]] = cir.extract_member %[[RESULT]][7] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[SIX_1:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:    %[[OUT_ADDR6:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SIX_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA6]], %[[OUT_ADDR6]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[OUT_DATA7:.+]] = cir.extract_member %[[RESULT]][8] : !rec_anon_struct1 -> !cir.vector<2 x !s64i>
  // CIR:    %[[SEVEN_1:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:    %[[OUT_ADDR7:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SEVEN_1]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[OUT_DATA7]], %[[OUT_ADDR7]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  } else {
  // CIR:    %[[NULL:.+]] = cir.const #cir.zero : !cir.vector<2 x !s64i>
  // CIR:    %[[ZERO_2:.+]] = cir.const #cir.int<0> : !u32i
  // CIR:    %[[OUT_ADDR0_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ZERO_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR0_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[ONE_2:.+]] = cir.const #cir.int<1> : !u32i
  // CIR:    %[[OUT_ADDR1_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[ONE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR1_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[TWO_2:.+]] = cir.const #cir.int<2> : !u32i
  // CIR:    %[[OUT_ADDR2_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[TWO_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR2_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[THREE_2:.+]] = cir.const #cir.int<3> : !u32i
  // CIR:    %[[OUT_ADDR3_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[THREE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR3_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[FOUR_2:.+]] = cir.const #cir.int<4> : !u32i
  // CIR:    %[[OUT_ADDR4_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FOUR_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR4_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[FIVE_2:.+]] = cir.const #cir.int<5> : !u32i
  // CIR:    %[[OUT_ADDR5_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[FIVE_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR5_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[SIX_2:.+]] = cir.const #cir.int<6> : !u32i
  // CIR:    %[[OUT_ADDR6_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SIX_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR6_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    %[[SEVEN_2:.+]] = cir.const #cir.int<7> : !u32i
  // CIR:    %[[OUT_ADDR7_1:.+]] = cir.ptr_stride %[[ODATA_ADDR]], %[[SEVEN_2]] : (!cir.ptr<!cir.vector<2 x !s64i>>, !u32i) -> !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:    cir.store align(16) %[[NULL]], %[[OUT_ADDR7_1]] : !cir.vector<2 x !s64i>, !cir.ptr<!cir.vector<2 x !s64i>>
  // CIR:  }
  // CIR:  %[[FLAG1:.+]] = cir.extract_member %[[RESULT]][0] : !rec_anon_struct1 -> !u8i
  // CIR:  cir.store %[[FLAG1]], %[[RET_ADDR:.+]] : !u8i, !cir.ptr<!u8i>
  // CIR:  %[[RET:.+]] = cir.load %[[RET_ADDR]] : !cir.ptr<!u8i>, !u8i
  // CIR:  cir.return %[[RET]] : !u8i

  // LLVM-LABEL: _mm_aesdecwide128kl_u8
  // LLVM:   %[[IN_DATA0:.+]] = load <2 x i64>, ptr %[[IDATA_ADDR:.+]], align 16
  // LLVM:   %[[IN_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 1
  // LLVM:   %[[IN_DATA1:.+]] = load <2 x i64>, ptr %[[IN_ADDR1]], align 16
  // LLVM:   %[[IN_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 2
  // LLVM:   %[[IN_DATA2:.+]] = load <2 x i64>, ptr %[[IN_ADDR2]], align 16
  // LLVM:   %[[IN_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 3
  // LLVM:   %[[IN_DATA3:.+]] = load <2 x i64>, ptr %[[IN_ADDR3]], align 16
  // LLVM:   %[[IN_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 4
  // LLVM:   %[[IN_DATA4:.+]] = load <2 x i64>, ptr %[[IN_ADDR4]], align 16
  // LLVM:   %[[IN_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 5
  // LLVM:   %[[IN_DATA5:.+]] = load <2 x i64>, ptr %[[IN_ADDR5]], align 16
  // LLVM:   %[[IN_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 6
  // LLVM:   %[[IN_DATA6:.+]] = load <2 x i64>, ptr %[[IN_ADDR6]], align 16
  // LLVM:   %[[IN_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i64 7
  // LLVM:   %[[IN_DATA7:.+]] = load <2 x i64>, ptr %[[IN_ADDR7]], align 16
  // LLVM:   %[[RESULT:.+]] = call { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.aesdecwide128kl(ptr %[[H_ADDR:.+]], <2 x i64>  %[[IN_DATA0]], <2 x i64>  %[[IN_DATA1]], <2 x i64>  %[[IN_DATA2]], <2 x i64>  %[[IN_DATA3]], <2 x i64>  %[[IN_DATA4]], <2 x i64>  %[[IN_DATA5]], <2 x i64>  %[[IN_DATA6]], <2 x i64>  %[[IN_DATA7]])
  // LLVM:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // LLVM:   %[[SUCC:.+]] = trunc i8 %34 to i1
  // LLVM:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // LLVM: [[NO_ERROR]]:
  // LLVM:   %[[OUT_DATA0:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 1
  // LLVM:   store <2 x i64> %[[OUT_DATA0]], ptr %[[ODATA_PTR:.+]], align 16
  // LLVM:   %[[OUT_DATA1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 2
  // LLVM:   %[[OUT_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 1
  // LLVM:   store <2 x i64> %[[OUT_DATA1]], ptr %[[OUT_ADDR1]], align 16
  // LLVM:   %[[OUT_DATA2:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 3
  // LLVM:   %[[OUT_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 2
  // LLVM:   store <2 x i64> %[[OUT_DATA2]], ptr %[[OUT_ADDR2]], align 16
  // LLVM:   %[[OUT_DATA3:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 4
  // LLVM:   %[[OUT_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 3
  // LLVM:   store <2 x i64> %[[OUT_DATA3]], ptr %[[OUT_ADDR3]], align 16
  // LLVM:   %[[OUT_DATA4:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 5
  // LLVM:   %[[OUT_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 4
  // LLVM:   store <2 x i64> %[[OUT_DATA4]], ptr %[[OUT_ADDR4]], align 16
  // LLVM:   %[[OUT_DATA5:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 6
  // LLVM:   %[[OUT_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 5
  // LLVM:   store <2 x i64> %[[OUT_DATA5]], ptr %[[OUT_ADDR5]], align 16
  // LLVM:   %[[OUT_DATA6:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 7
  // LLVM:   %[[OUT_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 6
  // LLVM:   store <2 x i64> %[[OUT_DATA6]], ptr %[[OUT_ADDR6]], align 16
  // LLVM:   %[[OUT_DATA7:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 8
  // LLVM:   %[[OUT_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 7
  // LLVM:   store <2 x i64> %[[OUT_DATA7]], ptr %[[OUT_ADDR7]], align 16
  // LLVM:   br label %[[EXIT:.+]]
  // LLVM: [[ERROR]]:
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // LLVM:   %[[OUT_ADDR0_1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 1
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_1]], align 16
  // LLVM:   %[[OUT_ADDR0_2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 2
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_2]], align 16
  // LLVM:   %[[OUT_ADDR0_3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 3
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_3]], align 16
  // LLVM:   %[[OUT_ADDR0_4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 4
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_4]], align 16
  // LLVM:   %[[OUT_ADDR0_5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 5
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_5]], align 16
  // LLVM:   %[[OUT_ADDR0_6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 6
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_6]], align 16
  // LLVM:   %[[OUT_ADDR0_7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i64 7
  // LLVM:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_7]], align 16
  // LLVM:   br label %[[EXIT]]
  // LLVM: [[EXIT]]:
  // LLVM:   %[[FLAG1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // LLVM:   store i8 %[[FLAG1]], ptr %[[FLAG1_ADDR:.+]], align 1
  // LLVM:   %[[FLAG2:.+]] = load i8, ptr %[[FLAG1_ADDR]], align 1
  // LLVM:   store i8 %[[FLAG2:.+]], ptr %[[FLAG2_ADDR:.+]], align 1
  // LLVM:   %[[RET:.+]] = load i8, ptr %[[FLAG2_ADDR]], align 1
  // LLVM:   ret i8 %[[RET]]

  // OGCG-LABEL: _mm_aesdecwide128kl_u8
  // OGCG:   %[[IN_DATA0:.+]] = load <2 x i64>, ptr %[[IDATA_ADDR:.+]], align 16
  // OGCG:   %[[IN_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 1
  // OGCG:   %[[IN_DATA1:.+]] = load <2 x i64>, ptr %[[IN_ADDR1]], align 16
  // OGCG:   %[[IN_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 2
  // OGCG:   %[[IN_DATA2:.+]] = load <2 x i64>, ptr %[[IN_ADDR2]], align 16
  // OGCG:   %[[IN_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 3
  // OGCG:   %[[IN_DATA3:.+]] = load <2 x i64>, ptr %[[IN_ADDR3]], align 16
  // OGCG:   %[[IN_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 4
  // OGCG:   %[[IN_DATA4:.+]] = load <2 x i64>, ptr %[[IN_ADDR4]], align 16
  // OGCG:   %[[IN_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 5
  // OGCG:   %[[IN_DATA5:.+]] = load <2 x i64>, ptr %[[IN_ADDR5]], align 16
  // OGCG:   %[[IN_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 6
  // OGCG:   %[[IN_DATA6:.+]] = load <2 x i64>, ptr %[[IN_ADDR6]], align 16
  // OGCG:   %[[IN_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[IDATA_ADDR]], i32 7
  // OGCG:   %[[IN_DATA7:.+]] = load <2 x i64>, ptr %[[IN_ADDR7]], align 16
  // OGCG:   %[[RESULT:.+]] = call { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.x86.aesdecwide128kl(ptr %[[H_ADDR:.+]], <2 x i64>  %[[IN_DATA0]], <2 x i64>  %[[IN_DATA1]], <2 x i64>  %[[IN_DATA2]], <2 x i64>  %[[IN_DATA3]], <2 x i64>  %[[IN_DATA4]], <2 x i64>  %[[IN_DATA5]], <2 x i64>  %[[IN_DATA6]], <2 x i64>  %[[IN_DATA7]])
  // OGCG:   %[[FLAG:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // OGCG:   %[[SUCC:.+]] = trunc i8 %[[FLAG]] to i1
  // OGCG:   br i1 %[[SUCC]], label %[[NO_ERROR:.+]], label %[[ERROR:.+]]
  // OGCG: [[NO_ERROR]]:
  // OGCG:   %[[OUT_DATA0:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 1
  // OGCG:   store <2 x i64> %[[OUT_DATA0]], ptr %[[ODATA_PTR:.+]], align 16
  // OGCG:   %[[OUT_DATA1:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 2
  // OGCG:   %[[OUT_ADDR1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 1
  // OGCG:   store <2 x i64> %[[OUT_DATA1]], ptr %[[OUT_ADDR1]], align 16
  // OGCG:   %[[OUT_DATA2:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 3
  // OGCG:   %[[OUT_ADDR2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 2
  // OGCG:   store <2 x i64> %[[OUT_DATA2]], ptr %[[OUT_ADDR2]], align 16
  // OGCG:   %[[OUT_DATA3:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 4
  // OGCG:   %[[OUT_ADDR3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 3
  // OGCG:   store <2 x i64> %[[OUT_DATA3]], ptr %[[OUT_ADDR3]], align 16
  // OGCG:   %[[OUT_DATA4:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 5
  // OGCG:   %[[OUT_ADDR4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 4
  // OGCG:   store <2 x i64> %[[OUT_DATA4]], ptr %[[OUT_ADDR4]], align 16
  // OGCG:   %[[OUT_DATA5:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 6
  // OGCG:   %[[OUT_ADDR5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 5
  // OGCG:   store <2 x i64> %[[OUT_DATA5]], ptr %[[OUT_ADDR5]], align 16
  // OGCG:   %[[OUT_DATA6:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 7
  // OGCG:   %[[OUT_ADDR6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 6
  // OGCG:   store <2 x i64> %[[OUT_DATA6]], ptr %[[OUT_ADDR6]], align 16
  // OGCG:   %[[OUT_DATA7:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 8
  // OGCG:   %[[OUT_ADDR7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 7
  // OGCG:   store <2 x i64> %[[OUT_DATA7]], ptr %[[OUT_ADDR7]], align 16
  // OGCG:   br label %[[EXIT:.+]]
  // OGCG: [[ERROR]]:
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[ODATA_PTR]], align 16
  // OGCG:   %[[OUT_ADDR0_1:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 1
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_1]], align 16
  // OGCG:   %[[OUT_ADDR0_2:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 2
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_2]], align 16
  // OGCG:   %[[OUT_ADDR0_3:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 3
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_3]], align 16
  // OGCG:   %[[OUT_ADDR0_4:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 4
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_4]], align 16
  // OGCG:   %[[OUT_ADDR0_5:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 5
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_5]], align 16
  // OGCG:   %[[OUT_ADDR0_6:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 6
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_6]], align 16
  // OGCG:   %[[OUT_ADDR0_7:.+]] = getelementptr <2 x i64>, ptr %[[ODATA_PTR]], i32 7
  // OGCG:   store <2 x i64> zeroinitializer, ptr %[[OUT_ADDR0_7]], align 16
  // OGCG:   br label %[[EXIT]]
  // OGCG: [[EXIT]]:
  // OGCG:   %[[RET:.+]] = extractvalue { i8, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } %[[RESULT]], 0
  // OGCG:   ret i8 %[[RET]]
  return _mm_aesdecwide128kl_u8(odata, idata, h);
}
