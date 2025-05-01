// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s -o - | FileCheck %s
// CHECK: llvm.store volatile %{{.+}}, %{{.+}} : i32, !llvm.ptr
// CHECK: %{{.+}} = llvm.load volatile %{{.+}} : !llvm.ptr -> i32
func.func @foo() {
  %true = arith.constant true
  %false = arith.constant false
  %0 = fir.alloca !fir.logical<4> {bindc_name = "a", uniq_name = "_QFEa"}
  %1 = fir.volatile_cast %0 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>, volatile>
  %2 = fir.alloca !fir.logical<4> {bindc_name = "b", uniq_name = "_QFEb"}
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %4 = fir.convert %false : (i1) -> !fir.logical<4>
  fir.store %4 to %1 : !fir.ref<!fir.logical<4>, volatile>
  %5 = fir.load %1 : !fir.ref<!fir.logical<4>, volatile>
  fir.store %5 to %2 : !fir.ref<!fir.logical<4>>
  %6 = fir.convert %true : (i1) -> !fir.logical<4>
  fir.store %6 to %1 : !fir.ref<!fir.logical<4>, volatile>
  return
}
