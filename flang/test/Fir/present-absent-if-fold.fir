// RUN: fir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @fold_present_absent_if_box(
// CHECK-SAME:     %[[VAR:.*]]: !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {
// CHECK-NOT: fir.if
// CHECK-NOT: fir.is_present
// CHECK-NOT: fir.absent
// CHECK: return %[[VAR]] : !fir.box<!fir.array<?xf32>>
func.func @fold_present_absent_if_box(
    %var: !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {
  %present = fir.is_present %var : (!fir.box<!fir.array<?xf32>>) -> i1
  %if_result = fir.if %present -> (!fir.box<!fir.array<?xf32>>) {
    fir.result %var : !fir.box<!fir.array<?xf32>>
  } else {
    %absent = fir.absent !fir.box<!fir.array<?xf32>>
    fir.result %absent : !fir.box<!fir.array<?xf32>>
  }
  return %if_result : !fir.box<!fir.array<?xf32>>
}

// CHECK-LABEL: func.func @fold_present_absent_if_ref(
// CHECK-SAME:     %[[VAR:.*]]: !fir.ref<i32>) -> !fir.ref<i32> {
// CHECK-NOT: fir.if
// CHECK: return %[[VAR]] : !fir.ref<i32>
func.func @fold_present_absent_if_ref(%var: !fir.ref<i32>) -> !fir.ref<i32> {
  %present = fir.is_present %var : (!fir.ref<i32>) -> i1
  %if_result = fir.if %present -> (!fir.ref<i32>) {
    fir.result %var : !fir.ref<i32>
  } else {
    %absent = fir.absent !fir.ref<i32>
    fir.result %absent : !fir.ref<i32>
  }
  return %if_result : !fir.ref<i32>
}

// CHECK-LABEL: func.func @fold_present_if_absent_csed(
// CHECK-SAME:     %[[VAR:.*]]: !fir.ref<i32>) -> !fir.ref<i32> {
// CHECK-NOT: fir.if
// CHECK: return %[[VAR]] : !fir.ref<i32>
func.func @fold_present_if_absent_csed(%var: !fir.ref<i32>) -> !fir.ref<i32> {
  %absent = fir.absent !fir.ref<i32>
  %present = fir.is_present %var : (!fir.ref<i32>) -> i1
  %if_result = fir.if %present -> (!fir.ref<i32>) {
    fir.result %var : !fir.ref<i32>
  } else {
    fir.result %absent : !fir.ref<i32>
  }
  return %if_result : !fir.ref<i32>
}

func.func private @side_effect() -> ()

// CHECK-LABEL: func.func @no_fold_call_in_then(
// CHECK: fir.is_present
// CHECK: fir.if
func.func @no_fold_call_in_then(%var: !fir.ref<i32>) -> !fir.ref<i32> {
  %present = fir.is_present %var : (!fir.ref<i32>) -> i1
  %if_result = fir.if %present -> (!fir.ref<i32>) {
    fir.call @side_effect() : () -> ()
    fir.result %var : !fir.ref<i32>
  } else {
    %absent = fir.absent !fir.ref<i32>
    fir.result %absent : !fir.ref<i32>
  }
  return %if_result : !fir.ref<i32>
}

// CHECK-LABEL: func.func @no_fold_call_in_else(
// CHECK: fir.is_present
// CHECK: fir.if
func.func @no_fold_call_in_else(%var: !fir.ref<i32>) -> !fir.ref<i32> {
  %present = fir.is_present %var : (!fir.ref<i32>) -> i1
  %if_result = fir.if %present -> (!fir.ref<i32>) {
    fir.result %var : !fir.ref<i32>
  } else {
    %absent = fir.absent !fir.ref<i32>
    fir.call @side_effect() : () -> ()
    fir.result %absent : !fir.ref<i32>
  }
  return %if_result : !fir.ref<i32>
}


// CHECK-LABEL: func.func @no_fold_call_in_else_csed(
// CHECK: fir.is_present
// CHECK: fir.if
func.func @no_fold_call_in_else_csed(%var: !fir.ref<i32>) -> !fir.ref<i32> {
  %absent = fir.absent !fir.ref<i32>
  %present = fir.is_present %var : (!fir.ref<i32>) -> i1
  %if_result = fir.if %present -> (!fir.ref<i32>) {
    fir.result %var : !fir.ref<i32>
  } else {
    fir.call @side_effect() : () -> ()
    fir.result %absent : !fir.ref<i32>
  }
  return %if_result : !fir.ref<i32>
}
