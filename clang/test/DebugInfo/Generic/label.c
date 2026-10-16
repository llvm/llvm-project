// This test will test the correstness of generating DILabel and
// llvm.dbg.label for labels.
//
// RUN: %clang_cc1 %s -o - -emit-llvm -debug-info-kind=limited | FileCheck %s

int f1(int a, int b) {
  int sum;

top:
  // CHECK: #dbg_label([[LABEL_METADATA:!.*]],  [[LABEL_LOCATION:![0-9]+]]
  sum = a + b;
  return sum;
}

// CHECK: [[LABEL_METADATA]] = !DILabel({{.*}}, name: "top", {{.*}}, line: 9, column: 1)
// CHECK: [[LABEL_LOCATION]] = !DILocation(line: 9,
