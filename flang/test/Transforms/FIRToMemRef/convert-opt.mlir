/// Verify that fir.convert are only generated one per fir.declare and that
/// optional conversions are optimized appropriately. 
///
/// RUN: fir-opt --enable-fir-convert-opts %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:   func.func @load_scalar
// CHECK-COUNT-1: fir.convert
// CHECK-NOT:     fir.convert
func.func @load_scalar(%arg0: !fir.ref<f32>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "a"} :
       (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %2 = fir.load %1 : !fir.ref<f32>
  %3 = fir.load %1 : !fir.ref<f32>
  return
}

// CHECK-LABEL:   func.func @load_array1d
// CHECK-COUNT-1: fir.convert
// CHECK-NOT:     fir.convert
func.func @load_array1d(%arg0: !fir.ref<!fir.array<3xf32>>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%shape) dummy_scope %0 {uniq_name = "a"} :
       (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, !fir.dscope) ->
       !fir.ref<!fir.array<3xf32>>
  %2 = fir.array_coor %1(%shape) %c1 :
       (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %3 = fir.load %2 : !fir.ref<f32>
  %4 = fir.array_coor %1(%shape) %c1 :
       (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %5 = fir.load %4 : !fir.ref<f32>
  return
}

// CHECK-LABEL:   func.func @store_scalar
// CHECK-COUNT-1: fir.convert
// CHECK-NOT:     fir.convert
func.func @store_scalar(%arg0: !fir.ref<i32>) {
  %c7_i32 = arith.constant 7 : i32
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "a"} :
       (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  fir.store %c7_i32 to %1 : !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  return
}

// CHECK-LABEL:   func.func @store_array1d
// CHECK-COUNT-1: fir.convert
// CHECK-NOT:     fir.convert
func.func @store_array1d(%arg0: !fir.ref<!fir.array<3xi32>>) {
  %c1 = arith.constant 1 : index
  %c7_i32 = arith.constant 7 : i32
  %c3 = arith.constant 3 : index
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %1 = fir.declare %arg0(%shape) dummy_scope %0 {uniq_name = "a"} :
       (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, !fir.dscope) ->
       !fir.ref<!fir.array<3xi32>>
  %2 = fir.array_coor %1(%shape) %c1 :
       (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  fir.store %c7_i32 to %2 : !fir.ref<i32>
  %3 = fir.array_coor %1(%shape) %c1 :
       (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  return
}

// When Present() checks the same optional memref than the one accessed inside
// the if statement, the convert is hoisted near the if statement.
// CHECK-LABEL: func.func @optional_optimized
// CHECK: [[DECLARE0:%[0-9]]] = fir.declare
// CHECK: [[DECLARE:%[0-9]]] = fir.declare %arg0
// CHECK: [[PRESENT:%[0-9]]] = fir.is_present [[DECLARE]]
// CHECK-NEXT: scf.if [[PRESENT]]
// CHECK: [[BOXADDR:%[0-9]+]] = fir.box_addr [[DECLARE]]
// CHECK: [[CONVERT:%[0-9]+]] = fir.convert [[BOXADDR]] : (!fir.ref<!fir.array<?x!fir.logical<4>>>) -> memref<?xi32>
func.func @optional_optimized(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.optional}) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %0 = fir.undefined !fir.dscope
  %1 = fir.alloca i32 {uniq_name = "i"}
  %2 = fir.declare %1 {uniq_name = "i"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {uniq_name = "r"} :
       (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) ->
       !fir.box<!fir.array<?x!fir.logical<4>>>
  %6 = fir.is_present %3 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i1
  scf.if %6 {
    %7 = arith.index_cast %c1 : index to i32
    %c1_0 = arith.constant 1 : index
    %8 = arith.addi %c3, %c1_0 : index
    %9 = scf.for %arg1 = %c1 to %8 step %c1 iter_args(%arg2 = %7) -> (i32) {
      fir.store %arg2 to %2 : !fir.ref<i32>
      %10 = fir.load %2 : !fir.ref<i32>
      %11 = arith.extsi %10 : i32 to i64
      %box_addr = fir.box_addr %3 :
        (!fir.box<!fir.array<?x!fir.logical<4>>>) ->
        !fir.ref<!fir.array<?x!fir.logical<4>>>
      %lb, %extent, %stride = fir.box_dims %3, %c0 :
        (!fir.box<!fir.array<?x!fir.logical<4>>>, index) ->
        (index, index, index)
      %shape = fir.shape %extent : (index) -> !fir.shape<1>
      %12 = fir.array_coor %box_addr(%shape) %11 :
        (!fir.ref<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>, i64) ->
        !fir.ref<!fir.logical<4>>
      %13 = fir.load %12 : !fir.ref<!fir.logical<4>>
      %15 = arith.addi %arg1, %c1 : index
      %16 = fir.load %2 : !fir.ref<i32>
      %17 = arith.addi %16, %7 : i32
      scf.yield %17 : i32
    }
    fir.store %9 to %2 : !fir.ref<i32>
  } else {
  }
  return
}

// When optional memref access is not control dependent on a check of it, no
// hoisting is applied and the convert is placed closer to the load/store
// (inside the loop).
// CHECK-LABEL: func.func @optional
// CHECK: [[DECLARE1:%[0-9]]] = fir.declare %arg1
// CHECK: [[DECLARE2:%[0-9]]] = fir.declare
// CHECK: [[DECLARE0:%[0-9]]] = fir.declare %arg0
// CHECK: [[PRESENT:%[0-9]]] = fir.is_present [[DECLARE1]]
// CHECK: scf.if [[PRESENT]]
// CHECK: scf.for
// CHECK: [[BOXADDR:%.+]] = fir.box_addr [[DECLARE0]]
// CHECK: [[CONVERT:%.+]] = fir.convert [[BOXADDR]] : (!fir.ref<!fir.array<?x!fir.logical<4>>>) -> memref<?xi32>
func.func @optional(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.optional},
                    %arg1: !fir.ref<!fir.logical<4>> {fir.optional}) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg1 dummy_scope %0 {uniq_name = "d"} :
       (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
  %2 = fir.alloca i32 {uniq_name = "i"}
  %3 = fir.declare %2 {uniq_name = "i"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.declare %arg0 dummy_scope %0 {uniq_name = "r"} :
       (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) ->
       !fir.box<!fir.array<?x!fir.logical<4>>>
  %7 = fir.is_present %1 : (!fir.ref<!fir.logical<4>>) -> i1
  scf.if %7 {
    %8 = arith.index_cast %c1 : index to i32
    %c1_0 = arith.constant 1 : index
    %9 = arith.addi %c3, %c1_0 : index
    %10 = scf.for %arg2 = %c1 to %9 step %c1 iter_args(%arg3 = %8) -> (i32) {
      fir.store %arg3 to %3 : !fir.ref<i32>
      %11 = fir.load %3 : !fir.ref<i32>
      %12 = arith.extsi %11 : i32 to i64
      %box_addr = fir.box_addr %4 :
        (!fir.box<!fir.array<?x!fir.logical<4>>>) ->
        !fir.ref<!fir.array<?x!fir.logical<4>>>
      %lb, %extent, %stride = fir.box_dims %4, %c0 :
        (!fir.box<!fir.array<?x!fir.logical<4>>>, index) ->
        (index, index, index)
      %shape = fir.shape %extent : (index) -> !fir.shape<1>
      %13 = fir.array_coor %box_addr(%shape) %12 :
        (!fir.ref<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>, i64) ->
        !fir.ref<!fir.logical<4>>
      %14 = fir.load %13 : !fir.ref<!fir.logical<4>>
      %16 = arith.addi %arg2, %c1 : index
      %17 = fir.load %3 : !fir.ref<i32>
      %18 = arith.addi %17, %8 : i32
      scf.yield %18 : i32
    }
    fir.store %10 to %3 : !fir.ref<i32>
  } else {
  }
  return
}

// Derived from a real-world example: ensure that we only generate one convert
// for an absent optional argument and reuse it for multiple loads.
// CHECK-LABEL: func.func @optional_declare
// CHECK:       [[ABSENT:%[0-9]+]] = fir.absent !fir.ref<i32>
// CHECK:       [[DUMMY:%[0-9]+]] = fir.undefined !fir.dscope
// CHECK:       [[DECLARE0:%[0-9]+]] = fir.declare [[ABSENT]] dummy_scope [[DUMMY]]
// CHECK:       [[PRESENT:%[0-9]+]] = fir.is_present [[DECLARE0]]
// CHECK:       scf.if [[PRESENT]]
// CHECK:         [[CONVERT0:%.+]] = fir.convert [[DECLARE0]] : (!fir.ref<i32>) -> memref<i32>
// CHECK:         memref.load [[CONVERT0]][] : memref<i32>
func.func @optional_declare() {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = fir.absent !fir.ref<i32>
  %6 = fir.undefined !fir.dscope
  %7 = fir.declare %5 dummy_scope %6 {uniq_name = "_QFFtestEd"} :
       (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %9 = fir.is_present %7 : (!fir.ref<i32>) -> i1
  scf.if %9 {
    %10 = fir.load %7 : !fir.ref<i32>
    %11 = arith.cmpi eq, %10, %c0_i32 : i32
    scf.if %11 {
      %15 = fir.load %7 : !fir.ref<i32>
    }
  }
  return
}


