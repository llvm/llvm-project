// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm -o - %s | FileCheck %s

void while_test(int *List, int Length, int *List2, int Length2) {
  // CHECK: define {{.*}} @_Z10while_test
  int i = 0;

#pragma clang loop distribute(enable)
  while (i < Length) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
    List[i] = i * 2;
    i++;
  }

  i = 0;
  while (i < Length2) {
    // CHECK-NOT: br label {{.*}}, !llvm.loop
    List2[i] = i * 2;
    i++;
  }
}

// CHECK: ![[LOOP_1]] = distinct !{![[LOOP_1]], ![[DISTRIBUTE_ENABLE:.*]]}
// CHECK: ![[DISTRIBUTE_ENABLE]] = !{!"llvm.loop.distribute.enable", i1 true}
