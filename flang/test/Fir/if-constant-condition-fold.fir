// RUN: fir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @true_if_with_else(
// CHECK-SAME:     %[[X:[^:]*]]: i32, %[[Y:[^:]*]]: i32, %[[P:[^:]*]]: !fir.ref<i32>) {
// CHECK-NOT: fir.if
// CHECK: fir.store %[[X]] to %[[P]] : !fir.ref<i32>
// CHECK-NOT: fir.store %[[Y]]
func.func @true_if_with_else(%x: i32, %y: i32, %p: !fir.ref<i32>) {
  %true = arith.constant true
  fir.if %true {
    fir.store %x to %p : !fir.ref<i32>
  } else {
    fir.store %y to %p : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @true_if_results(
// CHECK-SAME:     %[[X:[^:]*]]: i32, %[[Y:[^:]*]]: i32) -> i32 {
// CHECK-NOT: fir.if
// CHECK: return %[[X]] : i32
func.func @true_if_results(%x: i32, %y: i32) -> i32 {
  %true = arith.constant true
  %r = fir.if %true -> (i32) {
    fir.result %x : i32
  } else {
    fir.result %y : i32
  }
  return %r : i32
}

// CHECK-LABEL: func.func @true_if_multiple_results(
// CHECK-SAME:     %[[X:[^:]*]]: i32, %[[Y:[^:]*]]: f32, %[[OTHER_X:[^:]*]]: i32, %[[OTHER_Y:[^:]*]]: f32) -> (i32, f32) {
// CHECK-NOT: fir.if
// CHECK: return %[[X]], %[[Y]] : i32, f32
func.func @true_if_multiple_results(
      %x: i32, %y: f32, %other_x: i32, %other_y: f32) -> (i32, f32) {
  %true = arith.constant true
  %r0, %r1 = fir.if %true -> (i32, f32) {
    fir.result %x, %y : i32, f32
  } else {
    fir.result %other_x, %other_y : i32, f32
  }
  return %r0, %r1 : i32, f32
}

// CHECK-LABEL: func.func @false_if_with_else(
// CHECK-SAME:     %[[X:[^:]*]]: i32, %[[Y:[^:]*]]: i32, %[[P:[^:]*]]: !fir.ref<i32>) {
// CHECK-NOT: fir.if
// CHECK: fir.store %[[Y]] to %[[P]] : !fir.ref<i32>
// CHECK-NOT: fir.store %[[X]]
func.func @false_if_with_else(%x: i32, %y: i32, %p: !fir.ref<i32>) {
  %false = arith.constant false
  fir.if %false {
    fir.store %x to %p : !fir.ref<i32>
  } else {
    fir.store %y to %p : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @false_if_results(
// CHECK-SAME:     %[[X:[^:]*]]: i32, %[[Y:[^:]*]]: i32) -> i32 {
// CHECK-NOT: fir.if
// CHECK: return %[[Y]] : i32
func.func @false_if_results(%x: i32, %y: i32) -> i32 {
  %false = arith.constant false
  %r = fir.if %false -> (i32) {
    fir.result %x : i32
  } else {
    fir.result %y : i32
  }
  return %r : i32
}

// CHECK-LABEL: func.func @false_if_no_else(
// CHECK-NOT: fir.if
// CHECK-NOT: fir.store
func.func @false_if_no_else(%x: i32, %p: !fir.ref<i32>) {
  %false = arith.constant false
  fir.if %false {
    fir.store %x to %p : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @nested_constant_if(
// CHECK-SAME:     %[[X:[^:]*]]: i32, %[[Y:[^:]*]]: i32, %[[Z:[^:]*]]: i32, %[[P:[^:]*]]: !fir.ref<i32>) {
// CHECK-NOT: fir.if
// CHECK: fir.store %[[Y]] to %[[P]] : !fir.ref<i32>
// CHECK-NOT: fir.store %[[X]]
// CHECK-NOT: fir.store %[[Z]]
func.func @nested_constant_if(%x: i32, %y: i32, %z: i32, %p: !fir.ref<i32>) {
  %true = arith.constant true
  %false = arith.constant false
  fir.if %true {
    fir.if %false {
      fir.store %x to %p : !fir.ref<i32>
    } else {
      fir.store %y to %p : !fir.ref<i32>
    }
  } else {
    fir.store %z to %p : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @nested_constant_if_else(
// CHECK-SAME:     %[[X:[^:]*]]: i32, %[[Y:[^:]*]]: i32, %[[Z:[^:]*]]: i32, %[[P:[^:]*]]: !fir.ref<i32>) {
// CHECK-NOT: fir.if
// CHECK: fir.store %[[X]] to %[[P]] : !fir.ref<i32>
// CHECK-NOT: fir.store %[[Y]]
// CHECK-NOT: fir.store %[[Z]]
func.func @nested_constant_if_else(%x: i32, %y: i32, %z: i32, %p: !fir.ref<i32>) {
  %true = arith.constant true
  %false = arith.constant false
  fir.if %false {
    fir.store %z to %p : !fir.ref<i32>
  } else {
    fir.if %true {
      fir.store %x to %p : !fir.ref<i32>
    } else {
      fir.store %y to %p : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL: func.func @nested_constant_if_results(
// CHECK-SAME:     %[[X:[^:]*]]: i32, %[[Y:[^:]*]]: i32, %[[Z:[^:]*]]: i32) -> i32 {
// CHECK-NOT: fir.if
// CHECK: return %[[Y]] : i32
func.func @nested_constant_if_results(%x: i32, %y: i32, %z: i32) -> i32 {
  %true = arith.constant true
  %false = arith.constant false
  %r = fir.if %true -> (i32) {
    %inner = fir.if %false -> (i32) {
      fir.result %x : i32
    } else {
      fir.result %y : i32
    }
    fir.result %inner : i32
  } else {
    fir.result %z : i32
  }
  return %r : i32
}

// CHECK-LABEL: func.func @nonconst_if(
// CHECK: fir.if
func.func @nonconst_if(%c: i1, %x: i32, %p: !fir.ref<i32>) {
  fir.if %c {
    fir.store %x to %p : !fir.ref<i32>
  }
  return
}
