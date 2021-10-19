// RUN: %clang_cc1 -emit-llvm -disable-llvm-passes -O -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow %s -o - -debug-info-kind=line-tables-only | FileCheck %s


void foo(volatile int a) {
  // CHECK: call void @llvm.ubsantrap(i8 0){{.*}} !dbg [[LOC:![0-9]+]]
  a = a + 1;
  a = a + 1;
}

// CHECK: [[LOC]] = !DILocation(line: 0
