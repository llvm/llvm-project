// RUN: fir-opt --split-input-file --fir-rematerialize-box-ops %s | FileCheck %s

// The dummy constants inside regions give FileCheck stable anchors for the
// rematerialized operations' insertion points.

func.func private @use_box_10(!fir.box<!fir.array<10xi32>>)
func.func private @use_box_dyn(!fir.box<!fir.array<?xi32>>)

// CHECK-LABEL: func.func @embox_mixed_uses(
func.func @embox_mixed_uses(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  // CHECK: %[[OUTER_BOX:.*]] = fir.embox
  %box = fir.embox %arg0(%shape) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  // CHECK: fir.call @use_box_10(%[[OUTER_BOX]])
  fir.call @use_box_10(%box) : (!fir.box<!fir.array<10xi32>>) -> ()
  // CHECK: omp.task {
  omp.task {
    // CHECK: arith.constant 0 : i32
    %c0 = arith.constant 0 : i32
    // CHECK-NEXT: %[[TASK_BOX:.*]] = fir.embox
    // CHECK-NEXT: fir.call @use_box_10(%[[TASK_BOX]])
    fir.call @use_box_10(%box) : (!fir.box<!fir.array<10xi32>>) -> ()
    omp.terminator
  }
  return
}

// -----

func.func private @use_box_10(!fir.box<!fir.array<10xi32>>)

// CHECK-LABEL: func.func @rebox_chain(
func.func @rebox_chain(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %box = fir.embox %arg0(%shape) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %rebox = fir.rebox %box : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<10xi32>>
  // CHECK: omp.task {
  omp.task {
    // CHECK: arith.constant 0 : i32
    %c0 = arith.constant 0 : i32
    // CHECK-NEXT: %[[TASK_BOX:.*]] = fir.embox
    // CHECK-NEXT: %[[TASK_REBOX:.*]] = fir.rebox %[[TASK_BOX]]
    // CHECK-NEXT: fir.call @use_box_10(%[[TASK_REBOX]])
    fir.call @use_box_10(%rebox) : (!fir.box<!fir.array<10xi32>>) -> ()
    omp.terminator
  }
  return
}

// -----

func.func private @use_box_10(!fir.box<!fir.array<10xi32>>)

// CHECK-LABEL: func.func @nested_regions(
func.func @nested_regions(%arg0: !fir.ref<!fir.array<10xi32>>) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %box = fir.embox %arg0(%shape) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  // CHECK: omp.parallel {
  omp.parallel {
    // CHECK: arith.constant 0 : i32
    %c0 = arith.constant 0 : i32
    // CHECK-NEXT: %[[PAR_BOX:.*]] = fir.embox
    // CHECK-NEXT: fir.call @use_box_10(%[[PAR_BOX]])
    fir.call @use_box_10(%box) : (!fir.box<!fir.array<10xi32>>) -> ()
    // CHECK: omp.task {
    omp.task {
      // CHECK: arith.constant 1 : i32
      %c1 = arith.constant 1 : i32
      // CHECK-NEXT: %[[TASK_BOX:.*]] = fir.embox
      // CHECK-NEXT: fir.call @use_box_10(%[[TASK_BOX]])
      fir.call @use_box_10(%box) : (!fir.box<!fir.array<10xi32>>) -> ()
      omp.terminator
    }
    omp.terminator
  }
  return
}

// -----

func.func private @use_box_dyn(!fir.box<!fir.array<?xi32>>)

// CHECK-LABEL: func.func @rebox_maybe_absent_box(
// CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>
func.func @rebox_maybe_absent_box(%arg0: !fir.box<!fir.array<?xi32>>) {
  %rebox = fir.rebox %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  // CHECK: omp.task {
  omp.task {
    // CHECK: arith.constant 0 : i32
    %c0 = arith.constant 0 : i32
    // CHECK-NEXT: %[[TASK_REBOX:.*]] = fir.rebox %[[ARG0]]
    // CHECK-NEXT: fir.call @use_box_dyn(%[[TASK_REBOX]])
    fir.call @use_box_dyn(%rebox) : (!fir.box<!fir.array<?xi32>>) -> ()
    omp.terminator
  }
  return
}

// -----

func.func private @use_class_none(!fir.class<none>)

// CHECK-LABEL: func.func @embox_maybe_absent_source_box(
// CHECK-SAME: %[[ARG0:.*]]: !fir.ref<none>, %[[SOURCE_BOX:.*]]: !fir.class<!fir.array<?xnone>>
func.func @embox_maybe_absent_source_box(%arg0: !fir.ref<none>, %arg1: !fir.class<!fir.array<?xnone>>) {
  %box = fir.embox %arg0 source_box %arg1 : (!fir.ref<none>, !fir.class<!fir.array<?xnone>>) -> !fir.class<none>
  // CHECK: omp.task {
  omp.task {
    // CHECK: arith.constant 0 : i32
    %c0 = arith.constant 0 : i32
    // CHECK-NEXT: %[[TASK_EMBOX:.*]] = fir.embox %[[ARG0]] source_box %[[SOURCE_BOX]]
    // CHECK-NEXT: fir.call @use_class_none(%[[TASK_EMBOX]])
    fir.call @use_class_none(%box) : (!fir.class<none>) -> ()
    omp.terminator
  }
  return
}

// -----

func.func private @use_box_none(!fir.box<none>)

// CHECK-LABEL: func.func @converted_embox_live_in(
func.func @converted_embox_live_in(%arg0: !fir.ref<!fir.array<?xi32>>, %n: index) {
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %box = fir.embox %arg0(%shape) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %converted = fir.convert %box : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  // CHECK: omp.task {
  omp.task {
    // CHECK: arith.constant 0 : i32
    %c0 = arith.constant 0 : i32
    // CHECK-NEXT: %[[TASK_BOX:.*]] = fir.embox
    // CHECK-NEXT: %[[TASK_CONVERT:.*]] = fir.convert %[[TASK_BOX]]
    // CHECK-NEXT: fir.call @use_box_none(%[[TASK_CONVERT]])
    fir.call @use_box_none(%converted) : (!fir.box<none>) -> ()
    omp.terminator
  }
  return
}
