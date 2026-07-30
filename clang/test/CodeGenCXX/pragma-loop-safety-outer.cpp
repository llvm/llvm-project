// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Verify that the outer loop has the inner loop's access in its
// llvm.loop.parallel_accesses property.
void vectorize_outer_test(int *List, int Length) {
#pragma clang loop vectorize(assume_safety) interleave(disable) unroll(disable)
  for (int i = 0; i < Length; i += 2) {
#pragma clang loop unroll(full)
    for (int j = 0; j < 2; j += 1)
      List[i + j] = (i + j) * 2;
  }
}

// CHECK: %[[MUL:.+]] = mul
// CHECK: store i32 %[[MUL]], ptr %{{.+}}, !llvm.access.group ![[ACCESS_GROUP_2:[0-9]+]]
// CHECK: br label %{{.+}}, !llvm.loop ![[INNER_LOOPID:[0-9]+]]
// CHECK: br label %{{.+}}, !llvm.loop ![[OUTER_LOOPID:[0-9]+]]

// CHECK: ![[ACCESS_GROUP_2]] = distinct !{}
// CHECK: ![[INNER_LOOPID]] = distinct !{![[INNER_LOOPID]],
// CHECK: ![[OUTER_LOOPID]] = distinct !{![[OUTER_LOOPID]], [[MP:![0-9]+]], ![[PARALLEL_ACCESSES_9:[0-9]+]]
// CHECK: ![[PARALLEL_ACCESSES_9]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_2]]}
