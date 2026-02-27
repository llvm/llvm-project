// RUN: fir-opt %s --fir-to-scf --allow-unregistered-dialect | FileCheck %s

// Test that compiler generated fir.do_loop with a result from an iter_arg
// via a nested fir.if is converted.
//
// derived from:
// subroutine reduce(a,r)
//   integer :: a(3,3), r(3)
//   r = sum(a,dim=1,mask =.true.)
// end

// CHECK-LABEL: func.func @reduce_
// CHECK:       [[FOR_RESULT:%[0-9]+]] = scf.for
// CHECK:       [[IF_RESULT:%[0-9]+]] = scf.if
// CHECK:       scf.yield
// CHECK-NEXT:  } else {
// CHECK-NEXT:  scf.yield
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.yield [[IF_RESULT]]
// CHECK:       fir.store [[FOR_RESULT]]

func.func @reduce_(%arg0: !fir.ref<!fir.array<3x3xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3xi32>> {fir.bindc_name = "r"}) attributes {fir.internal_name = "_QPreduce", noinline} {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %c3 = arith.constant 3 : index
  %0 = fir.undefined !fir.dscope
  %shape2 = fir.shape %c3, %c3 : (index, index) -> !fir.shape<2>
  %shape1 = fir.shape %c3 : (index) -> !fir.shape<1>
  %a = fir.declare %arg0(%shape2) dummy_scope %0 {uniq_name = "_QFreduceEa"} : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<3x3xi32>>
  %r = fir.declare %arg1(%shape1) dummy_scope %0 {uniq_name = "_QFreduceEr"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xi32>>
  fir.do_loop %arg2 = %c1 to %c3 step %c1 unordered {
    %3 = fir.do_loop %arg3 = %c1 to %c3 step %c1 unordered iter_args(%arg4 = %c0_i32) -> (i32) {
      %5 = fir.if %true -> (i32) {
        %6 = fir.array_coor %a(%shape2) %arg3, %arg2 : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
        %7 = fir.load %6 : !fir.ref<i32>
        %8 = arith.addi %arg4, %7 : i32
        fir.result %8 : i32
      } else {
        fir.result %arg4 : i32
      }
      fir.result %5 : i32
    }
    %4 = fir.array_coor %r(%shape1) %arg2 : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %3 to %4 : !fir.ref<i32>
  }
  return
}


