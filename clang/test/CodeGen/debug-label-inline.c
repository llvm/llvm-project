// This test will test the correctness of generating DILabel and
// llvm.dbg.label when the label is in inlined functions.
//
// RUN: %clang_cc1 -O2 %s -o - -emit-llvm -debug-info-kind=limited | FileCheck %s
inline int f1(int a, int b) {
  int sum;

top:
  sum = a + b;
  return sum;
}

extern int ga, gb;

int f2(void) {
  int result;

  result = f1(ga, gb);
  // CHECK: #dbg_label([[LABEL_METADATA:!.*]],  [[LABEL_LOCATION:![0-9]+]]

  return result;
}

// CHECK: distinct !DISubprogram(name: "f1", {{.*}}, retainedNodes: [[ELEMENTS:!.*]])
// CHECK: [[ELEMENTS]] = !{{{.*}}, [[LABEL_METADATA]]}
// CHECK: [[LABEL_METADATA]] = !DILabel({{.*}}, name: "top", {{.*}}, line: 8, column: 1)
// CHECK: [[INLINEDAT:!.*]] = distinct !DILocation(line: 18,
// CHECK: [[LABEL_LOCATION]] = !DILocation(line: 8, {{.*}}, inlinedAt: [[INLINEDAT]])
