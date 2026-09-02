// RUN: fir-opt --stack-arrays %s | FileCheck %s

// Test that stack allocations are created in the block where the enclosing
// construct expects them (fir::getAllocaBlock) and are not hoisted out of it:
// each concurrent execution of a construct modelling parallelism needs its own
// storage.

// The allocation has no operand but must still stay inside the compute
// construct instead of being hoisted to the function entry block.
func.func @acc_parallel_no_operand() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  acc.parallel {
    %mem = fir.allocmem !fir.array<42xi32>
    %ref = fir.convert %mem : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
    %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    fir.store %c0_i32 to %elt : !fir.ref<i32>
    fir.freemem %mem : !fir.heap<!fir.array<42xi32>>
    acc.yield
  }
  return
}
// CHECK-LABEL: func.func @acc_parallel_no_operand()
// CHECK-NOT:     fir.alloca
// CHECK:         acc.parallel {
// CHECK-NEXT:      fir.alloca !fir.array<42xi32>
// CHECK-NOT:     fir.allocmem
// CHECK-NOT:     fir.freemem

func.func @acc_serial_no_operand() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  acc.serial {
    %mem = fir.allocmem !fir.array<42xi32>
    %ref = fir.convert %mem : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
    %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    fir.store %c0_i32 to %elt : !fir.ref<i32>
    fir.freemem %mem : !fir.heap<!fir.array<42xi32>>
    acc.yield
  }
  return
}
// CHECK-LABEL: func.func @acc_serial_no_operand()
// CHECK-NOT:     fir.alloca
// CHECK:         acc.serial {
// CHECK-NEXT:      fir.alloca !fir.array<42xi32>

func.func @acc_kernels_no_operand() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  acc.kernels {
    %mem = fir.allocmem !fir.array<42xi32>
    %ref = fir.convert %mem : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
    %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    fir.store %c0_i32 to %elt : !fir.ref<i32>
    fir.freemem %mem : !fir.heap<!fir.array<42xi32>>
    acc.terminator
  }
  return
}
// CHECK-LABEL: func.func @acc_kernels_no_operand()
// CHECK-NOT:     fir.alloca
// CHECK:         acc.kernels {
// CHECK-NEXT:      fir.alloca !fir.array<42xi32>

// The extent operand is available before the compute construct: the allocation
// can only be hoisted up to the beginning of the construct.
func.func @acc_parallel_operand_outside(%n: index) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %size = arith.addi %n, %n : index
  acc.parallel {
    %mem = fir.allocmem !fir.array<?xi32>, %size
    %ref = fir.convert %mem : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %c0_i32 to %elt : !fir.ref<i32>
    fir.freemem %mem : !fir.heap<!fir.array<?xi32>>
    acc.yield
  }
  return
}
// CHECK-LABEL: func.func @acc_parallel_operand_outside(
// CHECK:         %[[SIZE:.*]] = arith.addi
// CHECK-NOT:     fir.alloca
// CHECK:         acc.parallel {
// CHECK-NEXT:      fir.alloca !fir.array<?xi32>, %[[SIZE]]

// The extent operand is defined inside the compute construct: the allocation is
// placed right after it, as it would be in a plain function.
func.func @acc_parallel_operand_inside(%n: index) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  acc.parallel {
    %size = arith.addi %n, %n : index
    %mem = fir.allocmem !fir.array<?xi32>, %size
    %ref = fir.convert %mem : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    fir.store %c0_i32 to %elt : !fir.ref<i32>
    fir.freemem %mem : !fir.heap<!fir.array<?xi32>>
    acc.yield
  }
  return
}
// CHECK-LABEL: func.func @acc_parallel_operand_inside(
// CHECK-NOT:     fir.alloca
// CHECK:         acc.parallel {
// CHECK-NEXT:      %[[SIZE:.*]] = arith.addi
// CHECK-NEXT:      fir.alloca !fir.array<?xi32>, %[[SIZE]]

// Hoisting out of a sequential loop is still done, but only up to the beginning
// of the compute construct.
func.func @acc_parallel_sequential_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  acc.parallel {
    fir.do_loop %iv = %c1 to %c10 step %c1 {
      %mem = fir.allocmem !fir.array<42xi32>
      %ref = fir.convert %mem : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
      %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
      fir.store %c0_i32 to %elt : !fir.ref<i32>
      fir.freemem %mem : !fir.heap<!fir.array<42xi32>>
    }
    acc.yield
  }
  return
}
// CHECK-LABEL: func.func @acc_parallel_sequential_loop()
// CHECK-NOT:     fir.alloca
// CHECK:         acc.parallel {
// CHECK-NEXT:      fir.alloca !fir.array<42xi32>
// CHECK:           fir.do_loop
// CHECK-NOT:         fir.alloca

// An acc.loop is both loop-like and an automatic allocation scope: the
// allocation stays in it, and no stack save/restore is needed since the stack
// space is given back at the end of every iteration.
func.func @acc_loop_in_parallel() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  acc.parallel {
    acc.loop control(%iv : index) = (%c1 : index) to (%c10 : index) step (%c1 : index) {
      %mem = fir.allocmem !fir.array<42xi32>
      %ref = fir.convert %mem : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
      %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
      fir.store %c0_i32 to %elt : !fir.ref<i32>
      fir.freemem %mem : !fir.heap<!fir.array<42xi32>>
      acc.yield
    } inclusiveUpperbound(array<i1: true>) seq
    acc.yield
  }
  return
}
// CHECK-LABEL: func.func @acc_loop_in_parallel()
// CHECK-NOT:     fir.alloca
// CHECK:         acc.parallel {
// CHECK-NEXT:      acc.loop
// CHECK-NEXT:        fir.alloca !fir.array<42xi32>
// CHECK-NOT:     llvm.intr.stacksave
// CHECK-NOT:     llvm.intr.stackrestore

// Hoisting out of a sequential loop nested in an acc.loop removes the need for
// a stack save/restore.
func.func @sequential_loop_in_acc_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  acc.parallel {
    acc.loop control(%iv : index) = (%c1 : index) to (%c10 : index) step (%c1 : index) {
      fir.do_loop %jv = %c1 to %c10 step %c1 {
        %mem = fir.allocmem !fir.array<42xi32>
        %ref = fir.convert %mem : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
        %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
        fir.store %c0_i32 to %elt : !fir.ref<i32>
        fir.freemem %mem : !fir.heap<!fir.array<42xi32>>
      }
      acc.yield
    } inclusiveUpperbound(array<i1: true>) seq
    acc.yield
  }
  return
}
// CHECK-LABEL: func.func @sequential_loop_in_acc_loop()
// CHECK-NOT:     fir.alloca
// CHECK:         acc.parallel {
// CHECK-NEXT:      acc.loop
// CHECK-NEXT:        fir.alloca !fir.array<42xi32>
// CHECK:             fir.do_loop
// CHECK-NOT:           fir.alloca
// CHECK-NOT:     llvm.intr.stacksave
// CHECK-NOT:     llvm.intr.stackrestore

// acc.data is not a compute construct and does not own the allocations of its
// region: hoisting to the function entry block is allowed.
func.func @acc_data_no_operand() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  acc.data {
    %mem = fir.allocmem !fir.array<42xi32>
    %ref = fir.convert %mem : (!fir.heap<!fir.array<42xi32>>) -> !fir.ref<!fir.array<42xi32>>
    %elt = fir.coordinate_of %ref, %c0 : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    fir.store %c0_i32 to %elt : !fir.ref<i32>
    fir.freemem %mem : !fir.heap<!fir.array<42xi32>>
    acc.terminator
  } defaultAttr(none)
  return
}
// CHECK-LABEL: func.func @acc_data_no_operand()
// CHECK:         fir.alloca !fir.array<42xi32>
// CHECK:         acc.data
