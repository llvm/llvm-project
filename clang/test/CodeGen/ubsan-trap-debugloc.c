// RUN: %clang_cc1 -emit-llvm -disable-llvm-passes -O -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow %s -o - -debug-info-kind=line-tables-only | FileCheck %s


void foo(volatile int a) {
  // CHECK-LABEL: @foo
  // CHECK: call void @llvm.ubsantrap(i8 0){{.*}} !dbg [[LOC:![0-9]+]]
  a = a + 1;
  a = a + 1;
}

void bar(volatile int a) __attribute__((optnone)) {
  // CHECK-LABEL: @bar
  // CHECK: call void @llvm.ubsantrap(i8 0){{.*}} !dbg [[LOC2:![0-9]+]]
  // CHECK: call void @llvm.ubsantrap(i8 0){{.*}} !dbg [[LOC3:![0-9]+]]
  a = a + 1;
  a = a + 1;
}

// With optimisations enabled the traps are merged and need to share a debug location
// CHECK: [[LOC]] = !DILocation(line: 0

// With optimisations disabled the traps are not merged and retain accurate debug locations
// CHECK: [[LOC2]] = !DILocation(line: 15, column: 9
// CHECK: [[LOC3]] = !DILocation(line: 16, column: 9
