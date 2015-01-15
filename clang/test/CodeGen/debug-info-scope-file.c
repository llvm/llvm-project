// RUN: %clang_cc1 -g -emit-llvm < %s | FileCheck %s

// Check that, just because we emitted a function from a different file doesn't
// mean we insert a file-change inside the next function.

// CHECK: ret void, !dbg [[F1_LINE:![0-9]*]]
// CHECK: ret void, !dbg [[F2_LINE:![0-9]*]]
// CHECK: [[F1:![0-9]*]] = {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [def] [f1]
// CHECK: [[F2:![0-9]*]] = {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [def] [f2]
// CHECK: [[F1_LINE]] = !MDLocation({{.*}}, scope: [[F1]])
// CHECK: [[F2_LINE]] = !MDLocation({{.*}}, scope: [[F2]])

void f1() {
}

# 2 "foo.c"

void f2() {
}

