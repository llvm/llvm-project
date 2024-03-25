! Test lowering of intrinsic subroutines to HLFIR what matters here
! is not to test each subroutine, but to check how their
! lowering interfaces with the rest of lowering.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_subroutine(x)
 real :: x
 call cpu_time(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_subroutine(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}}
! CHECK:  %[[VAL_2:.*]] = fir.call @_FortranACpuTime() fastmath<contract> : () -> f64
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (f64) -> f32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_1]]#1 : !fir.ref<f32>

! test elemental subroutine calls
program main
  implicit none
  integer, parameter :: N = 3
  integer :: from(N), to(N)
  from = 7
  to = 6

  call mvbits(from, 2, 2, to, 0)
  if (any(to /= 5)) STOP 1
end program
! CHECK-LABEL:   func.func @_QQmain() attributes {fir.bindc_name = "main"} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "from", uniq_name = "_QFEfrom"}
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) {uniq_name = "_QFEfrom"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QFECn) : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFECn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "to", uniq_name = "_QFEto"}
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) {uniq_name = "_QFEto"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 7 : i32
! CHECK:           hlfir.assign %[[VAL_10]] to %[[VAL_3]]#0 : i32, !fir.ref<!fir.array<3xi32>>
! CHECK:           %[[VAL_11:.*]] = arith.constant 6 : i32
! CHECK:           hlfir.assign %[[VAL_11]] to %[[VAL_9]]#0 : i32, !fir.ref<!fir.array<3xi32>>
! CHECK:           %[[VAL_12:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_13:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_16:.*]] = %[[VAL_15]] to %[[VAL_0]] step %[[VAL_15]] {
! CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_16]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<i32>
! CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_16]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
! CHECK:             %[[VAL_21:.*]] = arith.constant 0 : i32
! CHECK:             %[[VAL_22:.*]] = arith.constant -1 : i32
! CHECK:             %[[VAL_23:.*]] = arith.constant 32 : i32
! CHECK:             %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_13]] : i32
! CHECK:             %[[VAL_25:.*]] = arith.shrui %[[VAL_22]], %[[VAL_24]] : i32
! CHECK:             %[[VAL_26:.*]] = arith.shli %[[VAL_25]], %[[VAL_14]] : i32
! CHECK:             %[[VAL_27:.*]] = arith.xori %[[VAL_26]], %[[VAL_22]] : i32
! CHECK:             %[[VAL_28:.*]] = arith.andi %[[VAL_27]], %[[VAL_20]] : i32
! CHECK:             %[[VAL_29:.*]] = arith.shrui %[[VAL_18]], %[[VAL_12]] : i32
! CHECK:             %[[VAL_30:.*]] = arith.andi %[[VAL_29]], %[[VAL_25]] : i32
! CHECK:             %[[VAL_31:.*]] = arith.shli %[[VAL_30]], %[[VAL_14]] : i32
! CHECK:             %[[VAL_32:.*]] = arith.ori %[[VAL_28]], %[[VAL_31]] : i32
! CHECK:             %[[VAL_33:.*]] = arith.cmpi eq, %[[VAL_13]], %[[VAL_21]] : i32
! CHECK:             %[[VAL_34:.*]] = arith.select %[[VAL_33]], %[[VAL_20]], %[[VAL_32]] : i32
! CHECK:             fir.store %[[VAL_34]] to %[[VAL_19]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           %[[VAL_35:.*]] = arith.constant 5 : i32
! CHECK:           %[[VAL_36:.*]] = hlfir.elemental %[[VAL_8]] unordered : (!fir.shape<1>) -> !hlfir.expr<3x!fir.logical<4>> {
! CHECK:           ^bb0(%[[VAL_37:.*]]: index):
! CHECK:             %[[VAL_38:.*]] = hlfir.designate %[[VAL_9]]#0 (%[[VAL_37]])  : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<i32>
! CHECK:             %[[VAL_40:.*]] = arith.cmpi ne, %[[VAL_39]], %[[VAL_35]] : i32
! CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i1) -> !fir.logical<4>
! CHECK:             hlfir.yield_element %[[VAL_41]] : !fir.logical<4>
! CHECK:           }
! CHECK:           %[[VAL_42:.*]] = hlfir.any %[[VAL_43:.*]] : (!hlfir.expr<3x!fir.logical<4>>) -> !fir.logical<4>
! CHECK:           hlfir.destroy %[[VAL_43]] : !hlfir.expr<3x!fir.logical<4>>
! CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_42]] : (!fir.logical<4>) -> i1
! CHECK:           cf.cond_br %[[VAL_44]], ^bb1, ^bb2
! CHECK:         ^bb1:
! CHECK:           %[[VAL_45:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_46:.*]] = arith.constant false
! CHECK:           %[[VAL_47:.*]] = arith.constant false
! CHECK:           %[[VAL_48:.*]] = fir.call @_FortranAStopStatement(%[[VAL_45]], %[[VAL_46]], %[[VAL_47]]) fastmath<contract> : (i32, i1, i1) -> none
! CHECK:           fir.unreachable
! CHECK:         ^bb2:
! CHECK:           return
! CHECK:         }
