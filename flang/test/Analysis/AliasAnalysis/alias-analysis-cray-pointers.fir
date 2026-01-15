// Check that cray pointers might alias with everything.

// RUN: fir-opt %s -unsafe-cray-pointers -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -mlir-disable-threading 2>&1 | FileCheck %s
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -mlir-disable-threading 2>&1 | FileCheck --check-prefix=DEFAULT %s

// Fortran source:
// subroutine test()
//   real :: a, b
//   pointer(p, a)
//   p = loc(b)
// end subroutine

// CHECK-LABEL: Testing : "_QPtest"
// CHECK-DAG: p#0 <-> b#0: MayAlias
// CHECK-DAG: p#1 <-> b#0: MayAlias
// CHECK-DAG: p#0 <-> b#1: MayAlias
// CHECK-DAG: p#1 <-> b#1: MayAlias
// CHECK-DAG: p#0 <-> a#0: MayAlias
// CHECK-DAG: p#1 <-> a#0: MayAlias
// CHECK-DAG: b#0 <-> a#0: MayAlias
// CHECK-DAG: b#1 <-> a#0: MayAlias
// CHECK-DAG: p#0 <-> a#1: MayAlias
// CHECK-DAG: p#1 <-> a#1: MayAlias
// CHECK-DAG: b#0 <-> a#1: MayAlias
// CHECK-DAG: b#1 <-> a#1: MayAlias

// By default, alias analysis assumes that cray pointers do not alias with
// non-target data. See flang/docs/Aliasing.md.
// DEFAULT-LABEL: Testing : "_QPtest"
// DEFAULT-DAG: p#0 <-> b#0: NoAlias
// DEFAULT-DAG: p#1 <-> b#0: NoAlias
// DEFAULT-DAG: p#0 <-> b#1: NoAlias
// DEFAULT-DAG: p#1 <-> b#1: NoAlias
// DEFAULT-DAG: p#0 <-> a#0: NoAlias
// DEFAULT-DAG: p#1 <-> a#0: NoAlias
// DEFAULT-DAG: b#0 <-> a#0: NoAlias
// DEFAULT-DAG: b#1 <-> a#0: NoAlias
// DEFAULT-DAG: p#0 <-> a#1: NoAlias
// DEFAULT-DAG: p#1 <-> a#1: NoAlias
// DEFAULT-DAG: b#0 <-> a#1: NoAlias
// DEFAULT-DAG: b#1 <-> a#1: NoAlias

func.func @_QPtest() {
  %0 = fir.alloca !fir.box<!fir.ptr<f32>>
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.alloca i64 {bindc_name = "p", uniq_name = "_QFtestEp"}
  %3:2 = hlfir.declare %2 {test.ptr = "p", fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFtestEp"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %4 = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFtestEb"}
  %5:2 = hlfir.declare %4 {test.ptr = "b", uniq_name = "_QFtestEb"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %6:2 = hlfir.declare %0 {test.ptr = "a", fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFtestEa"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %7 = fir.zero_bits !fir.ptr<f32>
  %8 = fir.embox %7 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.store %8 to %6#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %9 = fir.embox %5#0 : (!fir.ref<f32>) -> !fir.box<f32>
  %10 = fir.box_addr %9 : (!fir.box<f32>) -> !fir.ref<f32>
  %11 = fir.convert %10 : (!fir.ref<f32>) -> i64
  hlfir.assign %11 to %3#0 : i64, !fir.ref<i64>
  return
}

