// This test will test the correstness of generating DILabel and
// llvm.dbg.label for labels.
//
// RUN: %clang_cc1 -emit-llvm %s -o - -emit-llvm -debug-info-kind=limited | FileCheck %s

int f1(int a, int b) {
  int sum;

top:
  // CHECK: call void @llvm.dbg.label(metadata [[LABEL_METADATA:!.*]]), !dbg [[LABEL_LOCATION:!.*]]
  sum = a + b;
  return sum;
}

// CHECK: [[LABEL_METADATA]] = !DILabel({{.*}}, name: "top", {{.*}}, line: 9)
// CHECK: [[LABEL_LOCATION]] = !DILocation(line: 9,
