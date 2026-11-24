// RUN: fir-opt -flang-licm --split-input-file --mlir-print-debuginfo --mlir-print-local-scope %s | FileCheck %s

// Tests that LICM clears the line of a hoisted operation, keeping the file so
// consumers that require a file:line still have one. Operations it does not
// move keep their locations.

// A pure operation, hoisted because it is loop invariant.
func.func @_QPhoisted_pure(%arg0: !fir.ref<i32>, %arg1: i32, %n: index) {
  %c1 = arith.constant 1 : index
  fir.do_loop %i = %c1 to %n step %c1 {
    %0 = arith.muli %arg1, %arg1 : i32 loc("test.f90":14:5)
    fir.store %0 to %arg0 : !fir.ref<i32> loc("test.f90":14:5)
  } loc("test.f90":13:3)
  return loc("test.f90":16:1)
} loc("test.f90":12:1)

// CHECK-LABEL:   func.func @_QPhoisted_pure(
// CHECK:           %[[MUL:.*]] = arith.muli {{.*}} loc("test.f90":0:0)
// CHECK:           fir.do_loop
// CHECK:             fir.store %[[MUL]] to {{.*}} loc("test.f90":14:5)
// CHECK:           } loc("test.f90":13:3)
// CHECK:           return loc("test.f90":16:1)

// -----

// A load, hoisted because the loop does not modify what it reads and the
// location it reads is a non-OPTIONAL scalar dummy.
func.func @_QPhoisted_load(%arg0: !fir.ref<i32>, %n: index) {
  %c1 = arith.constant 1 : index
  %local = fir.alloca i32
  fir.do_loop %i = %c1 to %n step %c1 {
    %0 = fir.load %arg0 : !fir.ref<i32> loc("test.f90":24:5)
    fir.store %0 to %local : !fir.ref<i32> loc("test.f90":24:5)
  } loc("test.f90":23:3)
  return loc("test.f90":26:1)
} loc("test.f90":22:1)

// CHECK-LABEL:   func.func @_QPhoisted_load(
// CHECK:           %[[LOAD:.*]] = fir.load {{.*}} loc("test.f90":0:0)
// CHECK:           fir.do_loop
// CHECK:             fir.store %[[LOAD]] to {{.*}} loc("test.f90":24:5)
// CHECK:           } loc("test.f90":23:3)
// CHECK:           return loc("test.f90":26:1)

// -----

// Nothing is loop invariant here, so no location changes.
func.func @_QPnot_hoisted(%arg0: !fir.ref<i32>, %n: index) {
  %c1 = arith.constant 1 : index
  fir.do_loop %i = %c1 to %n step %c1 {
    %0 = fir.convert %i : (index) -> i32 loc("test.f90":34:5)
    fir.store %0 to %arg0 : !fir.ref<i32> loc("test.f90":34:5)
  } loc("test.f90":33:3)
  return loc("test.f90":36:1)
} loc("test.f90":32:1)

// CHECK-LABEL:   func.func @_QPnot_hoisted(
// CHECK:           fir.do_loop
// CHECK:             %[[CONV:.*]] = fir.convert {{.*}} loc("test.f90":34:5)
// CHECK:             fir.store %[[CONV]] to {{.*}} loc("test.f90":34:5)
// CHECK:           } loc("test.f90":33:3)
// CHECK:           return loc("test.f90":36:1)

// -----

// Composite locations are cleared in their leaves. A fused location keeps its
// metadata, and a callsite location keeps its caller so that the inlining
// context survives, if any.
func.func @_QPcomposite_locs(%arg0: !fir.ref<i32>, %arg1: i32, %n: index) {
  %c1 = arith.constant 1 : index
  fir.do_loop %i = %c1 to %n step %c1 {
    %0 = arith.muli %arg1, %arg1 : i32 loc(fused["test.f90":44:5, "test.f90":45:7])
    %1 = arith.addi %0, %arg1 : i32 loc(fused<"scope">["test.f90":44:5, "test.f90":45:7])
    %2 = arith.subi %1, %arg1 : i32 loc(callsite("test.f90":44:5 at "test.f90":48:3))
    fir.store %2 to %arg0 : !fir.ref<i32> loc("test.f90":44:5)
  } loc("test.f90":43:3)
  return loc("test.f90":46:1)
} loc("test.f90":42:1)

// CHECK-LABEL:   func.func @_QPcomposite_locs(
// CHECK:           %[[MUL:.*]] = arith.muli {{.*}} loc("test.f90":0:0)
// CHECK:           %[[ADD:.*]] = arith.addi {{.*}} loc(fused<"scope">["test.f90":0:0])
// CHECK:           %[[SUB:.*]] = arith.subi {{.*}} loc(callsite("test.f90":0:0 at "test.f90":48:3))
// CHECK:           fir.do_loop
// CHECK:             fir.store %[[SUB]] to {{.*}} loc("test.f90":44:5)
// CHECK:           } loc("test.f90":43:3)
// CHECK:           return loc("test.f90":46:1)
