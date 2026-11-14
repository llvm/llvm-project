// REQUIRES: x86-registered-target
// RUN: split-file %s %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O2 -emit-llvm %t/same.c -o - | FileCheck %s
// RUN: %clang_cc1 -triple i386-linux-gnu -O2 -emit-llvm %t/same.c -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -disable-llvm-passes -emit-llvm-bc %t/callee.c -o %t/callee.bc
// RUN: %clang_cc1 -triple x86_64-linux-gnu -O1 -disable-llvm-passes -emit-llvm-bc %t/caller.c -o %t/caller.bc
// RUN: llvm-lto -O2 -save-merged-module -exported-symbol=caller.default.1 -exported-symbol=caller.arch_haswell.0 -o %t/lto.o %t/caller.bc %t/callee.bc
// RUN: llvm-dis %t/lto.o.merged.bc -o - | FileCheck %s

// Matching CPU-model clone families should inline through the complete
// frontend/optimizer pipeline, both within a module and after LTO linking.
// CHECK-LABEL: define{{.*}} i32 @caller.default.1(
// CHECK-NOT: call
// CHECK: add nsw i32
// CHECK-NEXT: ret i32
// CHECK-LABEL: define{{.*}} i32 @caller.arch_haswell.0(
// CHECK-NOT: call
// CHECK: add nsw i32
// CHECK-NEXT: ret i32

//--- callee.c
__attribute__((target_clones("default,arch=haswell")))
int callee(int x) { return x + 1; }

//--- caller.c
int callee(int);
__attribute__((target_clones("default,arch=haswell")))
int caller(int x) { return callee(x); }

//--- same.c
#include "callee.c"
#include "caller.c"
