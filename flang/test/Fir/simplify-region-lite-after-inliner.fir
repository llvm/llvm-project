// RUN: tco %s | FileCheck %s


// In flang pipeline, the inliner calls createCanonicalizerPass with the region
// simplification disabled. The inliner pass does canonicalization even if
// no inlining happens. After canonicalization, FIR lite region simplification
// must be called to get rid of unreachable regions.
// This code exposes the need to run SimplifyRegionLitePass after the inliner is
// called with FIR pipeline.


func.func @repro(%arg0: i8, %arg1: i8) {
  %c34_i8 = arith.constant 34 : i8
  %c-1_i8 = arith.constant -1 : i8
  %2 = arith.xori %c34_i8, %c-1_i8 : i8
  %3 = arith.andi %arg0, %c34_i8 : i8
  %4 = arith.andi %arg1, %2 : i8
  %5 = arith.ori %3, %4 : i8
  %c34_i8_0 = arith.constant 34 : i8
  %7 = arith.andi %arg0, %c34_i8_0 : i8
  %c-35_i8 = arith.constant -35 : i8
  %9 = arith.andi %arg1, %c-35_i8 : i8
  %10 = arith.ori %7, %9 : i8
  %11 = arith.cmpi ne, %5, %10 : i8
  cf.cond_br %11, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %13 = func.call @foo() : () -> none
  cf.br ^bb2
^bb2:  // pred: ^bb0, ^bb2
  return
}
func.func private @foo() -> none


// CHECK: define void @repro(i8 %0, i8 %1)
// CHECK-NEXT: ret void
