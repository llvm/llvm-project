// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s --implicit-check-not='section "llvm.metadata"'
// RUN: %clang_cc1 %s -triple r600 -emit-llvm -o - | FileCheck %s --check-prefix AS1-GLOBALS
// END.

static __attribute((annotate("sfoo_0"))) __attribute((annotate("sfoo_1"))) char sfoo;
__attribute((annotate("foo_0"))) __attribute((annotate("foo_1"))) char foo;

void __attribute((annotate("ann_a_0"))) __attribute((annotate("ann_a_1"))) __attribute((annotate("ann_a_2"))) __attribute((annotate("ann_a_3"))) a(char *a);
void __attribute((annotate("ann_a_0"))) __attribute((annotate("ann_a_1"))) a(char *a) {
  __attribute__((annotate("bar_0"))) __attribute__((annotate("bar_1"))) static char bar;
  sfoo = 0;
}

__attribute((address_space(1))) __attribute__((annotate("addrspace1_ann"))) char addrspace1_var;

// CHECK: target triple
// CHECK-DAG: private unnamed_addr constant [7 x i8] c"sfoo_0\00", section "llvm.metadata"
// CHECK-DAG: private unnamed_addr constant [7 x i8] c"sfoo_1\00", section "llvm.metadata"

// CHECK-DAG: private unnamed_addr constant [6 x i8] c"foo_0\00", section "llvm.metadata"
// CHECK-DAG: private unnamed_addr constant [6 x i8] c"foo_1\00", section "llvm.metadata"

// CHECK-DAG: private unnamed_addr constant [8 x i8] c"ann_a_0\00", section "llvm.metadata"
// CHECK-DAG: private unnamed_addr constant [8 x i8] c"ann_a_1\00", section "llvm.metadata"
// CHECK-DAG: private unnamed_addr constant [8 x i8] c"ann_a_2\00", section "llvm.metadata"
// CHECK-DAG: private unnamed_addr constant [8 x i8] c"ann_a_3\00", section "llvm.metadata"

// CHECK-DAG: private unnamed_addr constant [6 x i8] c"bar_0\00", section "llvm.metadata"
// CHECK-DAG: private unnamed_addr constant [6 x i8] c"bar_1\00", section "llvm.metadata"

// CHECK-DAG: private unnamed_addr constant [15 x i8] c"addrspace1_ann\00", section "llvm.metadata"

// CHECK: @llvm.global.annotations = appending global [11 x { ptr, ptr, ptr, i32, ptr }] [{
// CHECK-SAME: { ptr @a.bar,
// CHECK-SAME: { ptr @a.bar,
// CHECK-SAME: { ptr @sfoo,
// CHECK-SAME: { ptr @sfoo,
// CHECK-SAME: { ptr @foo,
// CHECK-SAME: { ptr @foo,
// CHECK-SAME: { ptr addrspacecast (ptr addrspace(1) @addrspace1_var to ptr),
// CHECK-SAME: { ptr @a,
// CHECK-SAME: { ptr @a,
// CHECK-SAME: { ptr @a,
// CHECK-SAME: { ptr @a,
// CHECK-SAME: }], section "llvm.metadata"

// AS1-GLOBALS: target datalayout = "{{.+}}-A5-G1"
// AS1-GLOBALS: @llvm.global.annotations = appending addrspace(1) global [11 x { ptr addrspace(1), ptr addrspace(4), ptr addrspace(4), i32, ptr addrspace(4) }]
// AS1-GLOBALS-SAME: { ptr addrspace(1) @a.bar,
// AS1-GLOBALS-SAME: { ptr addrspace(1) @addrspace1_var,
