// RUN: fir-opt %s --fir-to-scf --allow-unregistered-dialect | FileCheck %s

// Test that compiler generated fir.do_loop with an result from an iter_arg
// is converted.
//
// derived from:
// subroutine test_any(a,b,res)
//   integer :: a(3), b(3)
//   logical :: res
//   res = any(a/=b)
// end

// CHECK-LABEL: func.func @test_any_
// CHECK:       [[SCF_RESULT:%[0-9]+]] = scf.for
// CHECK:       scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  [[CONVERT:%[0-9]+]] = fir.convert [[SCF_RESULT]]

func.func @test_any_(%arg0: !fir.ref<!fir.array<3xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xi32>> {fir.bindc_name = "b"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "res"}) attributes {fir.internal_name = "_QPtest_any"} {
  %c1 = arith.constant 1 : index
  %false = arith.constant false
  %c3 = arith.constant 3 : index
  %0 = fir.undefined !fir.dscope
  %1 = fircg.ext_declare %arg0(%c3) dummy_scope %0 {uniq_name = "_QFtest_anyEa"} : (!fir.ref<!fir.array<3xi32>>, index, !fir.dscope) -> !fir.ref<!fir.array<3xi32>>
  %2 = fircg.ext_declare %arg1(%c3) dummy_scope %0 {uniq_name = "_QFtest_anyEb"} : (!fir.ref<!fir.array<3xi32>>, index, !fir.dscope) -> !fir.ref<!fir.array<3xi32>>
  %3 = fircg.ext_declare %arg2 dummy_scope %0 {uniq_name = "_QFtest_anyEres"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> !fir.ref<!fir.logical<4>>
  %4 = fir.do_loop %arg3 = %c1 to %c3 step %c1 iter_args(%arg4 = %false) -> (i1) {
    %6 = fircg.ext_array_coor %1(%c3)<%arg3> : (!fir.ref<!fir.array<3xi32>>, index, index) -> !fir.ref<i32>
    %7 = fircg.ext_array_coor %2(%c3)<%arg3> : (!fir.ref<!fir.array<3xi32>>, index, index) -> !fir.ref<i32>
    %8 = fir.load %6 : !fir.ref<i32>
    %9 = fir.load %7 : !fir.ref<i32>
    %10 = arith.cmpi ne, %8, %9 : i32
    %11 = arith.ori %arg4, %10 : i1
    fir.result %11 : i1
  }
  %5 = fir.convert %4 : (i1) -> !fir.logical<4>
  fir.store %5 to %3 : !fir.ref<!fir.logical<4>>
  return
}


