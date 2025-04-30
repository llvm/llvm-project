// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

void test_loop_slot(int *List, int Length, int Value) {
#pragma ns mark slot("AABB")
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_1:.*]]
  }
}

void test_loop_multiple_marks(int *List, int Length, int Value) {
#pragma ns mark slot("ABCD") cgid("AA") duplication_count(3)
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_2:.*]]
  }
}

void test_loop_multiple_marks2(int *List, int Length, int Value) {
#pragma ns mark cgid("ABBC") slot("ABCABC") duplication_count(4)
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_3:.*]]
  }
}

void test_loop_cgid(int *List, int Length, int Value) {
#pragma ns mark cgid("AABB") duplication_count(6)
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_4:.*]]
  }
}

void test_loop_location_grid(int *List, int Length, int Value) {
#pragma ns location grid
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_5:.*]]
  }
}

void test_loop_location_risc(int *List, int Length, int Value) {
#pragma ns location risc
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_6:.*]]
  }
}

void test_loop_location_host(int *List, int Length, int Value) {
#pragma ns location host
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_7:.*]]
  }
}

void test_unmarked_loop_after_marked(int *List, int Length, int Value) {
#pragma ns location grid
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_8:.*]]
  }
  for (int i = 0; i < Length; i++) {
    List[i] *= Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_9:.*]]
  }
}

void test_loop_location_and_mark_together(int *List, int Length, int Value) {
#pragma ns location grid
#pragma ns mark cgid("AAA") duplication_count(8)
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_10:.*]]
  }
#pragma ns mark cgid("BBB") duplication_count(16)
#pragma ns location grid
  for (int i = 0; i < Length; i++) {
    List[i] = Value;
    // CHECK: br label {{.*}}, !llvm.loop ![[LOOP_11:.*]]
  }
}

// CHECK: ![[LOOP_1]] = distinct !{![[LOOP_1]], [[MUST_PROGRESS:.*]], ![[SLOT_AABB:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[SLOT_AABB]] = !{!"ns.loop.slot",  !"AABB"}

// CHECK: ![[LOOP_2]] = distinct !{![[LOOP_2]], [[MUST_PROGRESS:.*]], ![[SLOT_ABCD:.*]], ![[CGID_AA:.*]], ![[DUPLICATION_COUNT_3:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[SLOT_ABCD]] = !{!"ns.loop.slot",  !"ABCD"}
// CHECK: ![[CGID_AA]] = !{!"ns.loop.cgid",  !"AA"}
// CHECK: ![[DUPLICATION_COUNT_3]] = !{!"ns.loop.duplication_count", i32 3}

// CHECK: ![[LOOP_3]] = distinct !{![[LOOP_3]], [[MUST_PROGRESS:.*]], ![[SLOT_ABCABC:.*]], ![[CGID_ABBC:.*]], ![[DUPLICATION_COUNT_4:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[SLOT_ABCABC]] = !{!"ns.loop.slot",  !"ABCABC"}
// CHECK: ![[CGID_ABBC]] = !{!"ns.loop.cgid",  !"ABBC"}
// CHECK: ![[DUPLICATION_COUNT_4]] = !{!"ns.loop.duplication_count", i32 4}

// CHECK: ![[LOOP_4]] = distinct !{![[LOOP_4]], [[MUST_PROGRESS:.*]], ![[CGID_AABB:.*]], ![[DUPLICATION_COUNT_6:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[CGID_AABB]] = !{!"ns.loop.cgid",  !"AABB"}
// CHECK: ![[DUPLICATION_COUNT_6]] = !{!"ns.loop.duplication_count", i32 6}

// CHECK: ![[LOOP_5]] = distinct !{![[LOOP_5]], [[MUST_PROGRESS:.*]], ![[LOCATION_GRID:.*]], [[UNROLL_DISABLE:.*]]}

// CHECK: ![[LOOP_6]] = distinct !{![[LOOP_6]], [[MUST_PROGRESS:.*]], ![[LOCATION_RISC:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[LOCATION_RISC]] = !{!"ns.loop.location", !"risc"}

// CHECK: ![[LOOP_7]] = distinct !{![[LOOP_7]], [[MUST_PROGRESS:.*]], ![[LOCATION_HOST:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[LOCATION_HOST]] = !{!"ns.loop.location", !"host"}

// CHECK: ![[LOOP_8]] = distinct !{![[LOOP_8]], [[MUST_PROGRESS:.*]], ![[LOCATION_GRID:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[LOOP_9]] = distinct !{![[LOOP_9]], [[MUST_PROGRESS:.*]]}

// CHECK: ![[LOOP_10]] = distinct !{![[LOOP_10]], [[MUST_PROGRESS:.*]], ![[LOCATION_GRID:.*]], ![[CGID_AAA:.*]], ![[DUPLICATION_COUNT_8:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[CGID_AAA]] = !{!"ns.loop.cgid",  !"AAA"}
// CHECK: ![[DUPLICATION_COUNT_8]] = !{!"ns.loop.duplication_count", i32 8}

// CHECK: ![[LOOP_11]] = distinct !{![[LOOP_11]], [[MUST_PROGRESS:.*]], ![[LOCATION_GRID:.*]], ![[CGID_BBB:.*]], ![[DUPLICATION_COUNT_16:.*]], [[UNROLL_DISABLE:.*]]}
// CHECK: ![[CGID_BBB]] = !{!"ns.loop.cgid",  !"BBB"}
// CHECK: ![[DUPLICATION_COUNT_16]] = !{!"ns.loop.duplication_count", i32 16}
