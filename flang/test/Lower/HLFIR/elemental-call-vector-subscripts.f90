! Test passing of vector subscripted entities inside elemental
! procedures.
! RUN: bbc --emit-hlfir -o - %s | FileCheck %s

subroutine test()
  interface
    elemental subroutine foo(x, y)
      real, intent(in) :: x
      real, value :: y
    end subroutine
  end interface
  real :: x(10)
  call foo(x([1,3,7]), 0.)
end subroutine
! CHECK-LABEL:   func.func @_QPtest() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFtestEx"}
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) {uniq_name = "_QFtestEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QQro.3xi8.0) : !fir.ref<!fir.array<3xi64>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_6]])
! CHECK:           %[[VAL_8:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_11:.*]] = %[[VAL_10]] to %[[VAL_8]] step %[[VAL_10]] unordered {
! CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
! CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i64>
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:             fir.call @_QPfoo(%[[VAL_14]], %[[VAL_9]]) {{.*}}: (!fir.ref<f32>, f32) -> ()
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine test_value()
  interface
    elemental subroutine foo_value(x, y)
      real, value :: x
      real, value :: y
    end subroutine
  end interface
  real :: x(10)
  call foo_value(x([1,3,7]), 0.)
end subroutine

! CHECK-LABEL:   func.func @_QPtest_value() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "x", uniq_name = "_QFtest_valueEx"}
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_2]]) {uniq_name = "_QFtest_valueEx"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QQro.3xi8.0) : !fir.ref<!fir.array<3xi64>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_6]])
! CHECK:           %[[VAL_8:.*]] = arith.constant 3 : index
! CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
! CHECK:           ^bb0(%[[VAL_11:.*]]: index):
! CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_11]])  : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
! CHECK:             %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i64>
! CHECK:             %[[VAL_14:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
! CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<f32>
! CHECK:             hlfir.yield_element %[[VAL_15]] : f32
! CHECK:           }
! CHECK:           %[[VAL_16:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_18:.*]] = %[[VAL_17]] to %[[VAL_8]] step %[[VAL_17]] unordered {
! CHECK:             %[[VAL_19:.*]] = hlfir.apply %[[VAL_10]], %[[VAL_18]] : (!hlfir.expr<3xf32>, index) -> f32
! CHECK:             fir.call @_QPfoo_value(%[[VAL_19]], %[[VAL_16]]) {{.*}}: (f32, f32) -> ()
! CHECK:           }
! CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<3xf32>
! CHECK:           return

subroutine test_not_a_variable(i)
  interface
    elemental subroutine foo2(j)
      integer(8), intent(in) :: j
    end subroutine
  end interface
  integer(8) :: i(:)
  call foo2((i(i)))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_not_a_variable(
! CHECK:           hlfir.elemental
! CHECK:           %[[VAL_16:.*]] = hlfir.elemental
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_21:.*]] = {{.*}}
! CHECK:             %[[VAL_22:.*]] = hlfir.apply %[[VAL_16]], %[[VAL_21]] : (!hlfir.expr<?xi64>, index) -> i64
! CHECK:             %[[VAL_23:.*]]:3 = hlfir.associate %[[VAL_22]] {adapt.valuebyref} : (i64) -> (!fir.ref<i64>, !fir.ref<i64>, i1)
! CHECK:             fir.call @_QPfoo2(%[[VAL_23]]#1){{.*}}: (!fir.ref<i64>) -> ()
! CHECK:             hlfir.end_associate %[[VAL_23]]#1, %[[VAL_23]]#2 : !fir.ref<i64>, i1
! CHECK:           }
