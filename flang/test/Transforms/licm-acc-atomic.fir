// Test what the declared memory effects of acc.atomic.read permit around it.
// RUN: fir-opt %s --flang-licm | FileCheck %s
// RUN: fir-opt %s --loop-invariant-code-motion | FileCheck %s --check-prefix=GENERIC
// RUN: fir-opt %s -cse | FileCheck %s --check-prefix=CSE

// acc.atomic.read declares its effects per operand: it reads the location
// designated by `x` and writes the location designated by `v`. It declares
// nothing else, so it is modelled with relaxed ordering: the access to the
// location is atomic, but no ordering relationship is created with accesses to
// any other location, and the optimizer may move those across it.
//
// Without the declared effects the operation implements no
// MemoryEffectOpInterface at all. fir::AliasAnalysis::getModRef then falls into
// its `if (!interface) return getModAndRef()` branch, so the operation reads as
// "may modify everything" and no load in the enclosing loop can be hoisted,
// however unrelated it is.

// A load of a location the atomic neither reads nor writes is free to leave the
// loop. This is the case the declared effects unlock.

// CHECK-LABEL: func.func @hoist_unrelated_load_across_atomic
// CHECK:         fir.load
// CHECK:         scf.for
// CHECK-NOT:       fir.load
// CHECK:           acc.atomic.read
func.func @hoist_unrelated_load_across_atomic(
    %x: !fir.ref<i32>, %v: !fir.ref<i32>, %y: !fir.ref<i32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = fir.load %y : !fir.ref<i32>
    acc.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
  }
  return
}

// The effects are precise per operand rather than a blanket "touches memory":
// `v` is written, so a load of `v` is not loop invariant and stays put.

// CHECK-LABEL: func.func @no_hoist_load_of_atomic_destination
// CHECK:         scf.for
// CHECK:           fir.load
// CHECK:           acc.atomic.read
func.func @no_hoist_load_of_atomic_destination(
    %x: !fir.ref<i32>, %v: !fir.ref<i32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = fir.load %v : !fir.ref<i32>
    acc.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
  }
  return
}

// `x` is only read, so a load of `x` does not conflict with the atomic and is
// hoisted. Together with the previous function this pins that the two operands
// carry different effects.

// CHECK-LABEL: func.func @hoist_load_of_atomic_source
// CHECK:         fir.load
// CHECK:         scf.for
// CHECK-NOT:       fir.load
// CHECK:           acc.atomic.read
func.func @hoist_load_of_atomic_source(
    %x: !fir.ref<i32>, %v: !fir.ref<i32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %0 = fir.load %x : !fir.ref<i32>
    acc.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
  }
  return
}

// The generic MLIR loop-invariant-code-motion pass hoists on
// isMemoryEffectFree/isSpeculatable and consults no alias analysis, so it moves
// nothing here with or without the declared effects. Recorded so the contrast
// with flang-licm above is explicit.

// GENERIC-LABEL: func.func @hoist_unrelated_load_across_atomic
// GENERIC:         scf.for
// GENERIC:           fir.load
// GENERIC:           acc.atomic.read

// An atomic read is not removed by common subexpression elimination - it has
// declared effects, so two reads of the same location both survive.

// CSE-LABEL: func.func @cse_keeps_both_atomic_reads
// CSE:         acc.atomic.read
// CSE:         acc.atomic.read
func.func @cse_keeps_both_atomic_reads(%x: !fir.ref<i32>, %v: !fir.ref<i32>) {
  acc.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
  acc.atomic.read %v = %x : !fir.ref<i32>, !fir.ref<i32>, i32
  return
}
