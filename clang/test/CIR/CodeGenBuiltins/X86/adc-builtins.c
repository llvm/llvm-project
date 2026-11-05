// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefixes=CIR,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefixes=CIR,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefixes=LLVM,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefixes=LLVM,LLVM-X64 --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -emit-llvm -o - %s | FileCheck %s --check-prefixes=OGCG,OGCG-X64
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -emit-llvm -o - %s | FileCheck %s --check-prefixes=OGCG,OGCG-X64

// 32-bit x86 has only the _u32 forms.
// RUN: %clang_cc1 -x c -ffreestanding -triple i386-unknown-linux -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -ffreestanding -triple i386-unknown-linux -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -ffreestanding -triple i386-unknown-linux -emit-llvm -o - %s | FileCheck %s --check-prefix=OGCG

// This test mimics clang/test/CodeGen/X86/adc-builtins.c

#include <x86intrin.h>

unsigned char test_addcarry_u32(unsigned char __cf, unsigned int __x,
                                unsigned int __y, unsigned int *__p) {
  // CIR-LABEL: cir.func {{.*}}@{{.*}}test_addcarry_u32
  // CIR: cir.call @_addcarry_u32(
  // CIR-LABEL: cir.func {{.*}}@_addcarry_u32(
  // CIR: [[RES:%.*]] = cir.call_llvm_intrinsic "x86.addcarry.32"
  // CIR: [[VAL:%.*]] = cir.extract_member [[RES]][1] : {{.*}} -> !u32i
  // CIR: cir.store [[VAL]], {{%.*}} : !u32i, !cir.ptr<!u32i>
  // CIR: [[CF:%.*]] = cir.extract_member [[RES]][0] : {{.*}} -> !u8i
  // CIR: cir.store [[CF]], {{%.*}} : !u8i, !cir.ptr<!u8i>
  // LLVM-LABEL: define {{.*}}@{{.*}}test_addcarry_u32
  // LLVM: [[ADC:%.*]] = call { i8, i32 } @llvm.x86.addcarry.32
  // LLVM: [[DATA:%.*]] = extractvalue { i8, i32 } [[ADC]], 1
  // LLVM: store i32 [[DATA]], ptr {{%.*}}, align 4
  // LLVM: [[CF:%.*]] = extractvalue { i8, i32 } [[ADC]], 0
  // LLVM: store i8 [[CF]], ptr {{%.*}}, align 1
  // OGCG-LABEL: define {{.*}}@{{.*}}test_addcarry_u32
  // OGCG: [[ADC:%.*]] = call { i8, i32 } @llvm.x86.addcarry.32
  // OGCG: [[DATA:%.*]] = extractvalue { i8, i32 } [[ADC]], 1
  // OGCG: store i32 [[DATA]], ptr {{%.*}}, align 4
  // OGCG: [[CF:%.*]] = extractvalue { i8, i32 } [[ADC]], 0
  // OGCG: ret i8 [[CF]]
  return _addcarry_u32(__cf, __x, __y, __p);
}

#ifdef __x86_64__
unsigned char test_addcarry_u64(unsigned char __cf, unsigned long long __x,
                                unsigned long long __y,
                                unsigned long long *__p) {
  // CIR-X64-LABEL: cir.func {{.*}}@{{.*}}test_addcarry_u64
  // CIR-X64: cir.call @_addcarry_u64(
  // CIR-X64-LABEL: cir.func {{.*}}@_addcarry_u64(
  // CIR-X64: [[RES:%.*]] = cir.call_llvm_intrinsic "x86.addcarry.64"
  // CIR-X64: [[VAL:%.*]] = cir.extract_member [[RES]][1] : {{.*}} -> !u64i
  // CIR-X64: cir.store [[VAL]], {{%.*}} : !u64i, !cir.ptr<!u64i>
  // CIR-X64: [[CF:%.*]] = cir.extract_member [[RES]][0] : {{.*}} -> !u8i
  // CIR-X64: cir.store [[CF]], {{%.*}} : !u8i, !cir.ptr<!u8i>
  // LLVM-X64-LABEL: define {{.*}}@{{.*}}test_addcarry_u64
  // LLVM-X64: [[ADC:%.*]] = call { i8, i64 } @llvm.x86.addcarry.64
  // LLVM-X64: [[DATA:%.*]] = extractvalue { i8, i64 } [[ADC]], 1
  // LLVM-X64: store i64 [[DATA]], ptr {{%.*}}, align 8
  // LLVM-X64: [[CF:%.*]] = extractvalue { i8, i64 } [[ADC]], 0
  // LLVM-X64: store i8 [[CF]], ptr {{%.*}}, align 1
  // OGCG-X64-LABEL: define {{.*}}@{{.*}}test_addcarry_u64
  // OGCG-X64: [[ADC:%.*]] = call { i8, i64 } @llvm.x86.addcarry.64
  // OGCG-X64: [[DATA:%.*]] = extractvalue { i8, i64 } [[ADC]], 1
  // OGCG-X64: store i64 [[DATA]], ptr {{%.*}}, align 8
  // OGCG-X64: [[CF:%.*]] = extractvalue { i8, i64 } [[ADC]], 0
  // OGCG-X64: ret i8 [[CF]]
  return _addcarry_u64(__cf, __x, __y, __p);
}
#endif

unsigned char test_subborrow_u32(unsigned char __cf, unsigned int __x,
                                 unsigned int __y, unsigned int *__p) {
  // CIR-LABEL: cir.func {{.*}}@{{.*}}test_subborrow_u32
  // CIR: cir.call @_subborrow_u32(
  // CIR-LABEL: cir.func {{.*}}@_subborrow_u32(
  // CIR: [[RES:%.*]] = cir.call_llvm_intrinsic "x86.subborrow.32"
  // CIR: [[VAL:%.*]] = cir.extract_member [[RES]][1] : {{.*}} -> !u32i
  // CIR: cir.store [[VAL]], {{%.*}} : !u32i, !cir.ptr<!u32i>
  // CIR: [[CF:%.*]] = cir.extract_member [[RES]][0] : {{.*}} -> !u8i
  // CIR: cir.store [[CF]], {{%.*}} : !u8i, !cir.ptr<!u8i>
  // LLVM-LABEL: define {{.*}}@{{.*}}test_subborrow_u32
  // LLVM: [[SBB:%.*]] = call { i8, i32 } @llvm.x86.subborrow.32
  // LLVM: [[DATA:%.*]] = extractvalue { i8, i32 } [[SBB]], 1
  // LLVM: store i32 [[DATA]], ptr {{%.*}}, align 4
  // LLVM: [[CF:%.*]] = extractvalue { i8, i32 } [[SBB]], 0
  // LLVM: store i8 [[CF]], ptr {{%.*}}, align 1
  // OGCG-LABEL: define {{.*}}@{{.*}}test_subborrow_u32
  // OGCG: [[SBB:%.*]] = call { i8, i32 } @llvm.x86.subborrow.32
  // OGCG: [[DATA:%.*]] = extractvalue { i8, i32 } [[SBB]], 1
  // OGCG: store i32 [[DATA]], ptr {{%.*}}, align 4
  // OGCG: [[CF:%.*]] = extractvalue { i8, i32 } [[SBB]], 0
  // OGCG: ret i8 [[CF]]
  return _subborrow_u32(__cf, __x, __y, __p);
}

#ifdef __x86_64__
unsigned char test_subborrow_u64(unsigned char __cf, unsigned long long __x,
                                 unsigned long long __y,
                                 unsigned long long *__p) {
  // CIR-X64-LABEL: cir.func {{.*}}@{{.*}}test_subborrow_u64
  // CIR-X64: cir.call @_subborrow_u64(
  // CIR-X64-LABEL: cir.func {{.*}}@_subborrow_u64(
  // CIR-X64: [[RES:%.*]] = cir.call_llvm_intrinsic "x86.subborrow.64"
  // CIR-X64: [[VAL:%.*]] = cir.extract_member [[RES]][1] : {{.*}} -> !u64i
  // CIR-X64: cir.store [[VAL]], {{%.*}} : !u64i, !cir.ptr<!u64i>
  // CIR-X64: [[CF:%.*]] = cir.extract_member [[RES]][0] : {{.*}} -> !u8i
  // CIR-X64: cir.store [[CF]], {{%.*}} : !u8i, !cir.ptr<!u8i>
  // LLVM-X64-LABEL: define {{.*}}@{{.*}}test_subborrow_u64
  // LLVM-X64: [[SBB:%.*]] = call { i8, i64 } @llvm.x86.subborrow.64
  // LLVM-X64: [[DATA:%.*]] = extractvalue { i8, i64 } [[SBB]], 1
  // LLVM-X64: store i64 [[DATA]], ptr {{%.*}}, align 8
  // LLVM-X64: [[CF:%.*]] = extractvalue { i8, i64 } [[SBB]], 0
  // LLVM-X64: store i8 [[CF]], ptr {{%.*}}, align 1
  // OGCG-X64-LABEL: define {{.*}}@{{.*}}test_subborrow_u64
  // OGCG-X64: [[SBB:%.*]] = call { i8, i64 } @llvm.x86.subborrow.64
  // OGCG-X64: [[DATA:%.*]] = extractvalue { i8, i64 } [[SBB]], 1
  // OGCG-X64: store i64 [[DATA]], ptr {{%.*}}, align 8
  // OGCG-X64: [[CF:%.*]] = extractvalue { i8, i64 } [[SBB]], 0
  // OGCG-X64: ret i8 [[CF]]
  return _subborrow_u64(__cf, __x, __y, __p);
}
#endif
