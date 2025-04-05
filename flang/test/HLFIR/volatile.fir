// RUN: fir-opt --convert-hlfir-to-fir %s -o - | FileCheck %s
func.func @foo() {
  %true = arith.constant true
  %false = arith.constant false
  %0 = fir.alloca !fir.logical<4> {bindc_name = "a", uniq_name = "_QFEa"}
  %1 = fir.volatile_cast %0 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>, volatile>
  %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEa"} : (!fir.ref<!fir.logical<4>, volatile>) -> (!fir.ref<!fir.logical<4>, volatile>, !fir.ref<!fir.logical<4>, volatile>)
  %3 = fir.alloca !fir.logical<4> {bindc_name = "b", uniq_name = "_QFEb"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFEb"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %5 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.convert %false : (i1) -> !fir.logical<4>
  hlfir.assign %7 to %2#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
  %8 = fir.load %2#0 : !fir.ref<!fir.logical<4>, volatile>
  hlfir.assign %8 to %4#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  %9 = fir.convert %true : (i1) -> !fir.logical<4>
  hlfir.assign %9 to %2#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
  return
}
// CHECK: fir.store %{{.+}} to %{{.+}} : !fir.ref<!fir.logical<4>, volatile>
// CHECK: %{{.+}} = fir.load %{{.+}} : !fir.ref<!fir.logical<4>, volatile>
