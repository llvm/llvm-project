// UNSUPPORTED: target={{.*}}-zos{{.*}}
// RUN: %clang_cc1 %s -triple=riscv64 -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,RISCV64
// RUN: %clang_cc1 %s -triple=riscv64 -target-feature +d -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,RISCV64-D
// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT

// RISCV64: module asm(target_features: "+64bit,+i")
// RISCV64-D: module asm(target_features: "+64bit,+d,+f,+i,+zicsr")
// DEFAULT: module asm
// CHECK-NEXT: "foo1"
__asm__ ("foo1");
// CHECK-NEXT: "foo2"
__asm__ ("foo2");
// CHECK-NEXT: "foo3"
__asm__ ("foo3");
// CHECK-NEXT: "foo4"
__asm__ ("foo4");
// CHECK-NEXT: "foo5"
__asm__ ("foo5");
