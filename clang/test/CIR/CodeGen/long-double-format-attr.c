// This test mirrors CodeGen/long-double-module-flag.c
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-cir -o - | FileCheck %s --check-prefix=X86FP80-CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -DNOLDBL %s -emit-cir -o - | FileCheck %s --check-prefix=NOLDBL-CIR

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -fclangir -emit-llvm -o - | FileCheck %s --check-prefix=X86FP80
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu %s -fclangir -emit-llvm -o - | FileCheck %s --check-prefix=FP128

// FIXME: enable when arm abi is supported
// RUN-disabled: %clang_cc1 -triple arm-none-eabi %s -fclangir -emit-llvm -o - | FileCheck %s --check-prefix=DOUBLE
// FIXME: enable when PPC double-double format for long double is done
// RUN-disabled: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -fclangir -emit-llvm -o - | FileCheck %s --check-prefix=PPCFP128

// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -fclangir -emit-llvm -mabi=ieeelongdouble -o - | FileCheck %s --check-prefix=FP128
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -fclangir -emit-llvm -mlong-double-64 -o - | FileCheck %s --check-prefix=DOUBLE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -DNOLDBL %s -emit-llvm -o - | FileCheck %s --check-prefix=NOLDBL

// classic CG runs.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=X86FP80
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=FP128
// RUN: %clang_cc1 -triple arm-none-eabi %s -emit-llvm -o - | FileCheck %s --check-prefix=DOUBLE
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s --check-prefix=PPCFP128
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -emit-llvm -mabi=ieeelongdouble -o - | FileCheck %s --check-prefix=FP128
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -emit-llvm -mlong-double-64 -o - | FileCheck %s --check-prefix=DOUBLE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -DNOLDBL %s -emit-llvm -o - | FileCheck %s --check-prefix=NOLDBL

#ifndef NOLDBL
long double foo(long double a, long double b) {
  return a + b;
}
#endif

int bar() { return 1; }

// X86FP80-CIR: cir.long_double_format = #cir.long_double_format<x86_fp80>
// X86FP80: ![[#]] = !{i32 1, !"long-double-type", !"x86_fp80"}
// FP128: ![[#]] = !{i32 1, !"long-double-type", !"fp128"}
// DOUBLE: ![[#]] = !{i32 1, !"long-double-type", !"double"}
// PPCFP128: ![[#]] = !{i32 1, !"long-double-type", !"ppc_fp128"}
// NOLDBL-NOT: !"long-double-type"
// NOLDBL-CIR-NOT: cir.long_double_format
