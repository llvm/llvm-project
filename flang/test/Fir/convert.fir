// RUN: tco --target=x86_64-unknown-linux-gnu %s | FileCheck %s

// CHECK-LABEL: define { double, double } @c(<2 x float> %
func.func @c(%x : !fir.complex<4>) -> !fir.complex<8> {
// CHECK: %[[R:.*]] = extractvalue { float, float } %{{.*}}, 0
// CHECK: %[[I:.*]] = extractvalue { float, float } %{{.*}}, 1
// CHECK: %[[CR:.*]] = fpext float %[[R]] to double
// CHECK: %[[CI:.*]] = fpext float %[[I]] to double
// CHECK: %[[X:.*]] = insertvalue { double, double } undef, double %[[CR]], 0
// CHECK: insertvalue { double, double } %[[X]], double %[[CI]], 1
  %1 = fir.convert %x : (!fir.complex<4>) -> !fir.complex<8>
  return %1 : !fir.complex<8>
}
