// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

// Verify that the outer loop has the llvm.access.group property for the
// accesses outside and inside the inner loop, even when the inner loop
// is not perfectly nested.
void vectorize_imperfectly_nested_test(int *List, int Length) {
#pragma clang loop vectorize(assume_safety) interleave(disable) unroll(disable)
  for (int i = 0; i < Length; ++i) {
    List[i * Length] = 42;
#pragma clang loop vectorize(assume_safety) interleave(disable) unroll(disable)
    for (int j = 1; j < Length - 1; ++j)
      List[i * Length + j] = (i + j) * 2;
    List[(i + 1) * Length - 1] = 21;
  }
}


// CHECK: load i32, i32* %Length.addr, align 4, !llvm.access.group ![[ACCESS_GROUP_2:[0-9]+]]

// CHECK: %[[MUL:.+]] = mul nsw i32 %add, 2
// CHECK: store i32 %[[MUL]], i32* %{{.+}}, !llvm.access.group ![[ACCESS_GROUP_3:[0-9]+]]
// CHECK: br label %{{.+}}, !llvm.loop ![[INNER_LOOPID:[0-9]+]]
// CHECK: store i32 21, i32* %{{.+}}, !llvm.access.group ![[ACCESS_GROUP_2]]
// CHECK: br label %{{.+}}, !llvm.loop ![[OUTER_LOOPID:[0-9]+]]

// CHECK: ![[ACCESS_GROUP_2]] = distinct !{}
// CHECK: ![[ACCESS_GROUP_LIST_3:[0-9]+]]  = !{![[ACCESS_GROUP_2]], ![[ACCESS_GROUP_4:[0-9]+]]}
// CHECK: ![[ACCESS_GROUP_4]] = distinct !{}
// CHECK: ![[INNER_LOOPID]] = distinct !{![[INNER_LOOPID]], [[MP:![0-9]+]], ![[PARALLEL_ACCESSES_8:[0-9]+]]
// CHECK: [[MP]] = !{!"llvm.loop.mustprogress"}
// CHECK: ![[PARALLEL_ACCESSES_8]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_4]]}
// CHECK: ![[OUTER_LOOPID]] = distinct !{![[OUTER_LOOPID]], [[MP]], ![[PARALLEL_ACCESSES_10:[0-9]+]]
// CHECK: ![[PARALLEL_ACCESSES_10]] = !{!"llvm.loop.parallel_accesses", ![[ACCESS_GROUP_2]]}
