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
  // CHECK: call void @llvm.dbg.label(metadata [[LABEL_METADATA:!.*]]), !dbg [[LABEL_LOCATION:!.*]]

  return result;
}

// CHECK: distinct !DISubprogram(name: "f1", {{.*}}, retainedNodes: [[ELEMENTS:!.*]])
// CHECK: [[ELEMENTS]] = !{{{.*}}, [[LABEL_METADATA]]}
// CHECK: [[LABEL_METADATA]] = !DILabel({{.*}}, name: "top", {{.*}}, line: 8)
// CHECK: [[INLINEDAT:!.*]] = distinct !DILocation(line: 18,
// CHECK: [[LABEL_LOCATION]] = !DILocation(line: 8, {{.*}}, inlinedAt: [[INLINEDAT]])
