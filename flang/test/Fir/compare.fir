// RUN: fir-opt %s | tco --target=x86_64-unknown-linux-gnu | FileCheck %s

// CHECK-LABEL: define i1 @cmp3(<2 x float> %0, <2 x float> %1)
func.func @cmp3(%a : !fir.complex<4>, %b : !fir.complex<4>) -> i1 {
  // CHECK: fcmp oeq float
  %1 = fir.cmpc "oeq", %a, %b : !fir.complex<4>
  return %1 : i1
}

// CHECK-LABEL: define { double, double } @neg3(double %0, double %1)
func.func @neg3(%a : !fir.complex<8>) -> !fir.complex<8> {
  // CHECK: %[[g2:.*]] = insertvalue { double, double } %
  // CHECK: %[[r3:.*]] = fneg double
  // CHECK: insertvalue { double, double } %[[g2]], double %[[r3]]
  %1 = fir.negc %a : !fir.complex<8>
  return %1 : !fir.complex<8>
}
