// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefixes=CIR,CIR-X86 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefixes=CIR,CIR-X86 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefixes=LLVM,LLVM-X86 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefixes=LLVM,LLVM-X86 --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -emit-llvm -Wall -Werror %s -o - \
// RUN:   | FileCheck %s --check-prefixes=OGCG,OGCG-X86
// RUN: %clang_cc1 -x c++ -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:   -triple x86_64-unknown-linux -emit-llvm -Wall -Werror %s -o - \
// RUN:   | FileCheck %s --check-prefixes=OGCG,OGCG-X86

// This test copies clang/test/CodeGen/ms-barriers-intrinsics.c.

void _ReadWriteBarrier(void);
void _ReadBarrier(void);
void _WriteBarrier(void);
void __faststorefence(void);

void test_ReadWriteBarrier(void) { _ReadWriteBarrier(); }
// CIR-LABEL: test_ReadWriteBarrier
// CIR:  cir.atomic.fence syncscope(single_thread) seq_cst
// CIR:  cir.return

// LLVM-LABEL: test_ReadWriteBarrier
// LLVM: fence syncscope("singlethread") seq_cst
// LLVM: ret void

// OGCG-LABEL: test_ReadWriteBarrier
// OGCG: fence syncscope("singlethread") seq_cst
// OGCG: ret void

void test_ReadBarrier(void) { _ReadBarrier(); }
// CIR-LABEL: test_ReadBarrier
// CIR:  cir.atomic.fence syncscope(single_thread) seq_cst
// CIR:  cir.return

// LLVM-LABEL: test_ReadBarrier
// LLVM: fence syncscope("singlethread") seq_cst
// LLVM: ret void

// OGCG-LABEL: test_ReadBarrier
// OGCG: fence syncscope("singlethread") seq_cst
// OGCG: ret void

void test_WriteBarrier(void) { _WriteBarrier(); }
// CIR-LABEL: test_WriteBarrier
// CIR:  cir.atomic.fence syncscope(single_thread) seq_cst
// CIR:  cir.return

// LLVM-LABEL: test_WriteBarrier
// LLVM: fence syncscope("singlethread") seq_cst
// LLVM: ret void

// OGCG-LABEL: test_WriteBarrier
// OGCG: fence syncscope("singlethread") seq_cst
// OGCG: ret void

#if defined(__x86_64__)
void test__faststorefence(void) { __faststorefence(); }
// CIR-X86-LABEL: test__faststorefence
// CIR-X86:  cir.atomic.fence syncscope(system) seq_cst
// CIR-X86:  cir.return

// LLVM-X86-LABEL: test__faststorefence
// LLVM-X86: fence seq_cst
// LLVM-X86: ret void

// OGCG-X86-LABEL: test__faststorefence
// OGCG-X86: fence seq_cst
// OGCG-X86: ret void
#endif

