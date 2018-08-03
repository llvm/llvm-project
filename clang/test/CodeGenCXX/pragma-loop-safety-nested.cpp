// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Verify that the inner access is tagged with a parallel_loop_access
// for the inner and outer loop using a list.
void vectorize_nested_test(int *List, int Length) {
#pragma clang loop vectorize(assume_safety) interleave(disable) unroll(disable)
  for (int i = 0; i < Length; ++i) {
#pragma clang loop vectorize(assume_safety) interleave(disable) unroll(disable)
    for (int j = 0; j < Length; ++j)
      List[i * Length + j] = (i + j) * 2;
  }
}

// CHECK: %[[MUL:.+]] = mul
// CHECK: store i32 %[[MUL]], i32* %{{.+}}, !llvm.mem.parallel_loop_access ![[PARALLEL_LIST:[0-9]+]]
// CHECK: br label %{{.+}}, !llvm.loop ![[INNER_LOOPID:[0-9]+]]
// CHECK: br label %{{.+}}, !llvm.loop ![[OUTER_LOOPID:[0-9]+]]

// CHECK: ![[OUTER_LOOPID]] = distinct !{![[OUTER_LOOPID]],
// CHECK: ![[PARALLEL_LIST]] = !{![[OUTER_LOOPID]], ![[INNER_LOOPID]]}
// CHECK: ![[INNER_LOOPID]] = distinct !{![[INNER_LOOPID]],
