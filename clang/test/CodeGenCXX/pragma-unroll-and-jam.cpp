// RUN: %clang_cc1 -triple arm-none-eabi -std=c++11 -emit-llvm -o - %s | FileCheck %s

void unroll_and_jam(int *List, int Length, int Value) {
  // CHECK-LABEL: define {{.*}} @_Z14unroll_and_jam
#pragma unroll_and_jam
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_2:.*]]
      List[i * Length + j] = Value;
    }
  }
}

void unroll_and_jam_count(int *List, int Length, int Value) {
  // CHECK-LABEL: define {{.*}} @_Z20unroll_and_jam_count
#pragma unroll_and_jam(4)
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_3:.*]]
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_4:.*]]
      List[i * Length + j] = Value;
    }
  }
}

void nounroll_and_jam(int *List, int Length, int Value) {
  // CHECK-LABEL: define {{.*}} @_Z16nounroll_and_jam
#pragma nounroll_and_jam
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_5:.*]]
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_6:.*]]
      List[i * Length + j] = Value;
    }
  }
}

void clang_unroll_plus_nounroll_and_jam(int *List, int Length, int Value) {
  // CHECK-LABEL: define {{.*}} @_Z34clang_unroll_plus_nounroll_and_jam
#pragma nounroll_and_jam
#pragma unroll(4)
  for (int i = 0; i < Length; i++) {
    for (int j = 0; j < Length; j++) {
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_7:.*]]
      // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_8:.*]]
      List[i * Length + j] = Value;
    }
  }
}

// CHECK: ![[LOOP_2]] = distinct !{![[LOOP_2]], [[MP:![0-9]+]], ![[UNJ_ENABLE:.*]]}
// CHECK: ![[UNJ_ENABLE]] = !{!"llvm.loop.unroll_and_jam.enable"}
// CHECK: ![[LOOP_4]] = distinct !{![[LOOP_4]], [[MP]], ![[UNJ_4:.*]]}
// CHECK: ![[UNJ_4]] = !{!"llvm.loop.unroll_and_jam.count", i32 4}
// CHECK: ![[LOOP_6]] = distinct !{![[LOOP_6]], [[MP]], ![[UNJ_DISABLE:.*]]}
// CHECK: ![[UNJ_DISABLE]] = !{!"llvm.loop.unroll_and_jam.disable"}
// CHECK: ![[LOOP_8]] = distinct !{![[LOOP_8]], [[MP]], ![[UNJ_DISABLE:.*]], ![[UNROLL_4:.*]]}
// CHECK: ![[UNROLL_4]] = !{!"llvm.loop.unroll.count", i32 4}
