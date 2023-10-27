! Test lowering of nested array constructors as hlfir.elemental.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! hlfir.end_associate and hlfir.destroy used to be generated
! after hlfir.yield_element for the outermost hlfir.elemental.

! CHECK-LABEL:   func.func @_QPtest(
! CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "pi"},
! CHECK-SAME:                       %[[VAL_1:.*]]: !fir.ref<!fir.array<2xf32>> {fir.bindc_name = "h1"}) {
! CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_3]]) {uniq_name = "_QFtestEh1"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFtestEk"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFtestEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "l", uniq_name = "_QFtestEl"}
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFtestEl"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QFtestECn) : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtestECn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtestEpi"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           %[[VAL_12:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_14:.*]] = hlfir.elemental %[[VAL_13]] unordered : (!fir.shape<1>) -> !hlfir.expr<2xf32> {
! CHECK:           ^bb0(%[[VAL_15:.*]]: index):
! CHECK:             %[[VAL_16:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_17:.*]] = fir.shape %[[VAL_16]] : (index) -> !fir.shape<1>
! CHECK:             %[[VAL_18:.*]] = hlfir.elemental %[[VAL_17]] unordered : (!fir.shape<1>) -> !hlfir.expr<2xf32> {
! CHECK:             ^bb0(%[[VAL_19:.*]]: index):
! CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<f32>
! CHECK:               hlfir.yield_element %[[VAL_20]] : f32
! CHECK:             }
! CHECK:             %[[VAL_21:.*]]:3 = hlfir.associate %[[VAL_22:.*]](%[[VAL_17]]) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<2xf32>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>, i1)
! CHECK:             %[[VAL_23:.*]] = fir.embox %[[VAL_21]]#0(%[[VAL_17]]) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
! CHECK:             %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<!fir.array<?xf32>>
! CHECK:             %[[VAL_25:.*]] = fir.call @_QPfoo(%[[VAL_24]]) fastmath<contract> : (!fir.box<!fir.array<?xf32>>) -> f32
! CHECK:             hlfir.end_associate %[[VAL_21]]#1, %[[VAL_21]]#2 : !fir.ref<!fir.array<2xf32>>, i1
! CHECK:             hlfir.destroy %[[VAL_22]] : !hlfir.expr<2xf32>
! CHECK:             hlfir.yield_element %[[VAL_25]] : f32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_26:.*]] to %[[VAL_4]]#0 : !hlfir.expr<2xf32>, !fir.ref<!fir.array<2xf32>>
! CHECK:           hlfir.destroy %[[VAL_26]] : !hlfir.expr<2xf32>
! CHECK:           return
! CHECK:         }
subroutine test(pi,h1)
  implicit none
  integer, parameter :: N = 2
  interface
     pure real function foo(x)
       real, intent(in) :: x(:)
     end function foo
  end interface
  real h1(1:N)
  integer k, l
  real pi
  h1 = (/(foo((/(pi,l=1,N)/)),k=1,N)/)
end subroutine test
