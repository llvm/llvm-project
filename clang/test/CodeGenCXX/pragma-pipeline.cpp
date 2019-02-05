// RUN: %clang_cc1 -triple hexagon -std=c++11 -emit-llvm -o - %s | FileCheck %s

void pipeline_disabled(int *List, int Length, int Value) {
// CHECK-LABEL: define {{.*}} @_Z17pipeline_disabled
#pragma clang loop pipeline(disable)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
    List[i] = Value;
  }
}

void pipeline_not_disabled(int *List, int Length, int Value) {
  // CHECK-LABEL: define {{.*}} @_Z21pipeline_not_disabled
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
  }
}

void pipeline_initiation_interval(int *List, int Length, int Value) {
// CHECK-LABEL: define {{.*}} @_Z28pipeline_initiation_interval 
#pragma clang loop pipeline_initiation_interval(10)
  for (int i = 0; i < Length; i++) {
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_3:.*]]
    List[i] = Value;
  }
}

void pipeline_disabled_on_nested_loop(int *List, int Length, int Value) {
  // CHECK-LABEL: define {{.*}} @_Z32pipeline_disabled_on_nested_loop
  for (int i = 0; i < Length; i++) {
#pragma clang loop pipeline(disable)
    for (int j = 0; j < Length; j++) {
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_4:.*]]
      List[i * Length + j] = Value;
    }
  }
}

// CHECK: ![[LOOP_1]] = distinct !{![[LOOP_1]], ![[PIPELINE_DISABLE:.*]]}
// CHECK: ![[PIPELINE_DISABLE]] = !{!"llvm.loop.pipeline.disable", i1 true}

// CHECK-NOT:llvm.loop.pipeline

// CHECK: ![[LOOP_3]] = distinct !{![[LOOP_3]], ![[PIPELINE_II_10:.*]]}
// CHECK: ![[PIPELINE_II_10]] = !{!"llvm.loop.pipeline.initiationinterval", i32 10}

// CHECK: ![[LOOP_4]] = distinct !{![[LOOP_4]], ![[PIPELINE_DISABLE]]}
