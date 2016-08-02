// RUN: %clang_cc1 -debug-info-kind=limited -emit-llvm < %s | FileCheck %s

// Check that, just because we emitted a function from a different file doesn't
// mean we insert a file-change inside the next function.

// CHECK: ret void, !dbg [[F1_LINE:![0-9]*]]
// CHECK: ret void, !dbg [[F2_LINE:![0-9]*]]
// CHECK: [[F1:![0-9]*]] = distinct !DISubprogram(name: "f1",{{.*}} isDefinition: true
// CHECK: [[F2:![0-9]*]] = distinct !DISubprogram(name: "f2",{{.*}} isDefinition: true
// CHECK: [[F1_LINE]] = !DILocation({{.*}}, scope: [[F1]])
// CHECK: [[F2_LINE]] = !DILocation({{.*}}, scope: [[F2]])

void f1() {
}

# 2 "foo.c"

void f2() {
}

