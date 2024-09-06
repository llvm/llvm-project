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

func.func @convert_to_llvm_pointer(%x : !fir.llvm_ptr<i32>) -> !llvm.ptr {
  %0 = fir.convert %x : (!fir.llvm_ptr<i32>) -> !llvm.ptr
  return %0 : !llvm.ptr
}
// CHECK-LABEL: convert_to_llvm_pointer(
// CHECK-SAME: ptr %[[X:.*]])
// CHECK-NEXT: ret ptr %[[X]]

func.func @convert_from_llvm_pointer(%x : !llvm.ptr) -> !fir.llvm_ptr<i32> {
  %0 = fir.convert %x : (!llvm.ptr) -> !fir.llvm_ptr<i32>
  return %0 : !fir.llvm_ptr<i32>
}
// CHECK-LABEL: convert_from_llvm_pointer(
// CHECK-SAME: ptr %[[X:.*]])
// CHECK-NEXT: ret ptr %[[X]]
