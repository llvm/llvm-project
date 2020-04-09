// RUN: tco %s | FileCheck %s

// CHECK-LABEL: @ftest
func @ftest(%x : i1) -> i1 {
  // this pair of converts should be folded and DCEd
  %1 = fir.convert %x : (i1) -> !fir.logical<1>
  %2 = fir.convert %1 : (!fir.logical<1>) -> i1
  // CHECK-NEXT: ret i1 %0
  return %2 : i1
}

// CHECK-LABEL: @gtest
func @gtest(%x : !fir.logical<2>) -> !fir.logical<2> {
  // this pair of converts should be folded and DCEd
  %1 = fir.convert %x : (!fir.logical<2>) -> i1
  %2 = fir.convert %1 : (i1) -> !fir.logical<2>
  // CHECK-NEXT: ret i16 %0
  return %2 : !fir.logical<2>
}

// CHECK-LABEL: @htest
func @htest(%x : !fir.int<4>) -> !fir.int<4> {
  // these converts are NOPs and should be folded away
  %1 = fir.convert %x : (!fir.int<4>) -> !fir.int<4>
  %2 = fir.convert %1 : (!fir.int<4>) -> !fir.int<4>
  // CHECK-NEXT: ret i32 %0
  return %2 : !fir.int<4>
}
