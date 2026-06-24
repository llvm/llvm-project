// Check that cray pointers might alias with everything.

// RUN: fir-opt %s -unsafe-cray-pointers -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -mlir-disable-threading 2>&1 | FileCheck %s
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -mlir-disable-threading 2>&1 | FileCheck --check-prefix=DEFAULT %s

// Fortran source:
// subroutine test()
//   real :: a, b, c
//   pointer(p, a)
//   p = loc(b)
//   p = loc(c)
//   a = 1.0
// end subroutine

// CHECK-LABEL: Testing : "_QPtest"
// Legend:
// * 'a' is the descriptor address.
// * 'a.data" is the address of the 'a's data.
// * 'b' is the address of the 'b's data.
// * 'c' is the address of the 'c's data.
// * 'p' is the address of memory holding the raw address
//   of the 'b's data (or any other pointee's data)
// * 'raw_ptr_b' is the address of the 'b's data.
// * 'raw_ptr_c' is the address of the 'c's data.

// TODO: we should report NoAlias here, because
// 'p' is the address of location of the cray pointer value.
// It cannot alias with anything but itself.
// CHECK-DAG: p#0 <-> b#0: MayAlias
// CHECK-DAG: p#1 <-> b#0: MayAlias
// CHECK-DAG: p#0 <-> b#1: MayAlias
// CHECK-DAG: p#1 <-> b#1: MayAlias
// CHECK-DAG: p#0 <-> c#0: MayAlias
// CHECK-DAG: p#1 <-> c#0: MayAlias
// CHECK-DAG: p#0 <-> c#1: MayAlias
// CHECK-DAG: p#1 <-> c#1: MayAlias
// CHECK-DAG: p#0 <-> raw_ptr_b#0: MayAlias
// CHECK-DAG: p#1 <-> raw_ptr_b#0: MayAlias
// CHECK-DAG: p#0 <-> raw_ptr_c#0: MayAlias
// CHECK-DAG: p#1 <-> raw_ptr_c#0: MayAlias
// CHECK-DAG: p#0 <-> a.data#0: MayAlias
// CHECK-DAG: p#1 <-> a.data#0: MayAlias

// The descriptor address does not alias 'p'.
// CHECK-DAG: p#0 <-> a#0: NoAlias
// CHECK-DAG: p#1 <-> a#0: NoAlias
// CHECK-DAG: p#0 <-> a#1: NoAlias
// CHECK-DAG: p#1 <-> a#1: NoAlias

// 'b's data address and 'c's data address cannot alias.
// CHECK-DAG: b#0 <-> c#0: NoAlias
// CHECK-DAG: b#1 <-> c#0: NoAlias
// CHECK-DAG: b#0 <-> c#1: NoAlias
// CHECK-DAG: b#1 <-> c#1: NoAlias

// 'b's and 'c's data addresses cannot alias with
// the address of 'a' descriptor.
// CHECK-DAG: b#0 <-> a#0: NoAlias
// CHECK-DAG: b#1 <-> a#0: NoAlias
// CHECK-DAG: c#0 <-> a#0: NoAlias
// CHECK-DAG: c#1 <-> a#0: NoAlias
// CHECK-DAG: b#0 <-> a#1: NoAlias
// CHECK-DAG: b#1 <-> a#1: NoAlias
// CHECK-DAG: c#0 <-> a#1: NoAlias
// CHECK-DAG: c#1 <-> a#1: NoAlias

// 'b' == 'raw_ptr_b' and 'c' == 'raw_ptr_c'
// CHECK-DAG: b#0 <-> raw_ptr_b#0: MustAlias
// CHECK-DAG: b#1 <-> raw_ptr_b#0: MustAlias
// CHECK-DAG: c#0 <-> raw_ptr_c#0: MustAlias
// CHECK-DAG: c#1 <-> raw_ptr_c#0: MustAlias

// CHECK-DAG: c#0 <-> raw_ptr_b#0: NoAlias
// CHECK-DAG: c#1 <-> raw_ptr_b#0: NoAlias
// CHECK-DAG: a#0 <-> raw_ptr_b#0: NoAlias
// CHECK-DAG: a#1 <-> raw_ptr_b#0: NoAlias
// CHECK-DAG: b#0 <-> raw_ptr_c#0: NoAlias
// CHECK-DAG: b#1 <-> raw_ptr_c#0: NoAlias
// CHECK-DAG: a#0 <-> raw_ptr_c#0: NoAlias
// CHECK-DAG: a#1 <-> raw_ptr_c#0: NoAlias
// CHECK-DAG: raw_ptr_b#0 <-> raw_ptr_c#0: NoAlias

// 'a.data' may point to either 'b' or 'c'.
// CHECK-DAG: b#0 <-> a.data#0: MayAlias
// CHECK-DAG: b#1 <-> a.data#0: MayAlias
// CHECK-DAG: c#0 <-> a.data#0: MayAlias
// CHECK-DAG: c#1 <-> a.data#0: MayAlias
// CHECK-DAG: raw_ptr_b#0 <-> a.data#0: MayAlias
// CHECK-DAG: raw_ptr_c#0 <-> a.data#0: MayAlias

// The descriptor address 'a' does not alias the 'a's data.
// CHECK-DAG: a#0 <-> a.data#0: NoAlias
// CHECK-DAG: a#1 <-> a.data#0: NoAlias

// By default, alias analysis assumes that cray pointers do not alias with
// non-target data. See flang/docs/Aliasing.md.
// DEFAULT-LABEL: Testing : "_QPtest"
// NoAlias is always correct, though we report MayAlias
// under -unsafe-cray-pointers.
// DEFAULT-DAG: p#0 <-> b#0: NoAlias
// DEFAULT-DAG: p#1 <-> b#0: NoAlias
// DEFAULT-DAG: p#0 <-> b#1: NoAlias
// DEFAULT-DAG: p#1 <-> b#1: NoAlias
// DEFAULT-DAG: p#0 <-> c#0: NoAlias
// DEFAULT-DAG: p#1 <-> c#0: NoAlias
// DEFAULT-DAG: p#0 <-> c#1: NoAlias
// DEFAULT-DAG: p#1 <-> c#1: NoAlias
// DEFAULT-DAG: p#0 <-> raw_ptr_b#0: NoAlias
// DEFAULT-DAG: p#1 <-> raw_ptr_b#0: NoAlias
// DEFAULT-DAG: p#0 <-> raw_ptr_c#0: NoAlias
// DEFAULT-DAG: p#1 <-> raw_ptr_c#0: NoAlias
// DEFAULT-DAG: p#0 <-> a.data#0: NoAlias
// DEFAULT-DAG: p#1 <-> a.data#0: NoAlias

// Same as with -unsafe-cray-pointers.
// DEFAULT-DAG: p#0 <-> a#0: NoAlias
// DEFAULT-DAG: p#1 <-> a#0: NoAlias
// DEFAULT-DAG: p#0 <-> a#1: NoAlias
// DEFAULT-DAG: p#1 <-> a#1: NoAlias

// Same as with -unsafe-cray-pointers.
// DEFAULT-DAG: b#0 <-> c#0: NoAlias
// DEFAULT-DAG: b#1 <-> c#0: NoAlias
// DEFAULT-DAG: b#0 <-> c#1: NoAlias
// DEFAULT-DAG: b#1 <-> c#1: NoAlias

// Same as with -unsafe-cray-pointers.
// DEFAULT-DAG: b#0 <-> a#0: NoAlias
// DEFAULT-DAG: b#1 <-> a#0: NoAlias
// DEFAULT-DAG: c#0 <-> a#0: NoAlias
// DEFAULT-DAG: c#1 <-> a#0: NoAlias
// DEFAULT-DAG: b#0 <-> a#1: NoAlias
// DEFAULT-DAG: b#1 <-> a#1: NoAlias
// DEFAULT-DAG: c#0 <-> a#1: NoAlias
// DEFAULT-DAG: c#1 <-> a#1: NoAlias

// Same as with -unsafe-cray-pointers.
// DEFAULT-DAG: b#0 <-> raw_ptr_b#0: MustAlias
// DEFAULT-DAG: b#1 <-> raw_ptr_b#0: MustAlias
// DEFAULT-DAG: c#0 <-> raw_ptr_c#0: MustAlias
// DEFAULT-DAG: c#1 <-> raw_ptr_c#0: MustAlias

// Same as with -unsafe-cray-pointers.
// DEFAULT-DAG: c#0 <-> raw_ptr_b#0: NoAlias
// DEFAULT-DAG: c#1 <-> raw_ptr_b#0: NoAlias
// DEFAULT-DAG: a#0 <-> raw_ptr_b#0: NoAlias
// DEFAULT-DAG: a#1 <-> raw_ptr_b#0: NoAlias
// DEFAULT-DAG: b#0 <-> raw_ptr_c#0: NoAlias
// DEFAULT-DAG: b#1 <-> raw_ptr_c#0: NoAlias
// DEFAULT-DAG: a#0 <-> raw_ptr_c#0: NoAlias
// DEFAULT-DAG: a#1 <-> raw_ptr_c#0: NoAlias
// DEFAULT-DAG: raw_ptr_b#0 <-> raw_ptr_c#0: NoAlias

// This is the functional difference with -unsafe-cray-pointers.
// The safe assumption is MayAlias.
// DEFAULT-DAG: b#0 <-> a.data#0: NoAlias
// DEFAULT-DAG: b#1 <-> a.data#0: NoAlias
// DEFAULT-DAG: c#0 <-> a.data#0: NoAlias
// DEFAULT-DAG: c#1 <-> a.data#0: NoAlias
// DEFAULT-DAG: raw_ptr_b#0 <-> a.data#0: NoAlias
// DEFAULT-DAG: raw_ptr_c#0 <-> a.data#0: NoAlias

// Same as with -unsafe-cray-pointers.
// DEFAULT-DAG: a#0 <-> a.data#0: NoAlias
// DEFAULT-DAG: a#1 <-> a.data#0: NoAlias

func.func @_QPtest() {
  %0 = fir.alloca !fir.box<!fir.ptr<f32>>
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.alloca i64 {bindc_name = "p", uniq_name = "_QFtestEp"}
  %3:2 = hlfir.declare %2 {test.ptr = "p", fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFtestEp"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %4 = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFtestEb"}
  %5:2 = hlfir.declare %4 {test.ptr = "b", uniq_name = "_QFtestEb"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %6 = fir.alloca f32 {bindc_name = "c", uniq_name = "_QFtestEc"}
  %7:2 = hlfir.declare %6 {test.ptr = "c", uniq_name = "_QFtestEc"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %8:2 = hlfir.declare %0 {test.ptr = "a", fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFtestEa"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %9 = fir.zero_bits !fir.ptr<f32>
  %10 = fir.embox %9 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.store %10 to %8#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.embox %5#0 : (!fir.ref<f32>) -> !fir.box<f32>
  %12 = fir.box_addr %11 {test.ptr = "raw_ptr_b"} : (!fir.box<f32>) -> !fir.ref<f32>
  %13 = fir.convert %12 : (!fir.ref<f32>) -> i64
  hlfir.assign %13 to %3#0 : i64, !fir.ref<i64>
  %14 = fir.embox %7#0 : (!fir.ref<f32>) -> !fir.box<f32>
  %15 = fir.box_addr %14 {test.ptr = "raw_ptr_c"} : (!fir.box<f32>) -> !fir.ref<f32>
  %16 = fir.convert %15 : (!fir.ref<f32>) -> i64
  hlfir.assign %16 to %3#0 : i64, !fir.ref<i64>
  %cst = arith.constant 1.000000e+00 : f32
  // There should be _FortranAPointerAssociateScalar call here, but
  // we skip it for the test.
  %21 = fir.load %8#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %22 = fir.box_addr %21 {test.ptr = "a.data"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  hlfir.assign %cst to %22 : f32, !fir.ptr<f32>
  return
}

