! Test lowering of Constant<T>.
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_constant_scalar()
subroutine test_constant_scalar()
  print *, (10., 20.)
  ! CHECK-DAG:  %[[VAL_0:.*]] = arith.constant 2.000000e+01 : f32
  ! CHECK-DAG:  %[[VAL_1:.*]] = arith.constant 1.000000e+01 : f32
  ! CHECK:  %[[VAL_7:.*]] = fir.undefined complex<f32>
  ! CHECK:  %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_1]], [0 : index] : (complex<f32>, f32) -> complex<f32>
  ! CHECK:  %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_0]], [1 : index] : (complex<f32>, f32) -> complex<f32>
end subroutine

! CHECK-LABEL: func.func @_QPtest_constant_scalar_char()
subroutine test_constant_scalar_char()
  print *, "hello"
! CHECK:  %[[VAL_5:.*]] = fir.address_of(@[[name:.*]]) : !fir.ref<!fir.char<1,5>>
! CHECK:  %[[VAL_6:.*]] = arith.constant 5 : index
! CHECK:  hlfir.declare %[[VAL_5]] typeparams %[[VAL_6]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "[[name]]"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
end subroutine

! CHECK-LABEL: func.func @_QPtest_constant_array()
subroutine test_constant_array()
  print *, [1., 2., 3.]
! CHECK:  %[[VAL_5:.*]] = fir.address_of(@[[name:.*]]) : !fir.ref<!fir.array<3xf32>>
! CHECK:  %[[VAL_6:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:  hlfir.declare %[[VAL_5]](%[[VAL_7]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "[[name]]"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
end subroutine

! CHECK-LABEL: func.func @_QPtest_constant_array_char()
subroutine test_constant_array_char()
  print *, ["abc", "cde"]
! CHECK:  %[[VAL_5:.*]] = fir.address_of(@[[name:.*]]) : !fir.ref<!fir.array<2x!fir.char<1,3>>>
! CHECK:  %[[VAL_6:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_7:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:  hlfir.declare %[[VAL_5]](%[[VAL_8]]) typeparams %[[VAL_7]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "[[name]]"} : (!fir.ref<!fir.array<2x!fir.char<1,3>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<2x!fir.char<1,3>>>, !fir.ref<!fir.array<2x!fir.char<1,3>>>)
end subroutine

! CHECK-LABEL: func.func @_QPtest_constant_with_lower_bounds()
subroutine test_constant_with_lower_bounds()
  integer, parameter :: i(-1:0, -1:0) = reshape([1,2,3,4], shape=[2,2])
  print *, i
! CHECK:  %[[VAL_12:.*]] = fir.address_of(@_QFtest_constant_with_lower_boundsECi) : !fir.ref<!fir.array<2x2xi32>>
! CHECK:  %[[VAL_13:.*]] = arith.constant -1 : index
! CHECK:  %[[VAL_14:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_15:.*]] = arith.constant -1 : index
! CHECK:  %[[VAL_16:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_17:.*]] = fir.shape_shift %[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[VAL_16]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:  hlfir.declare %[[VAL_12]](%[[VAL_17]]) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_constant_with_lower_boundsECi"} : (!fir.ref<!fir.array<2x2xi32>>, !fir.shapeshift<2>) -> (!fir.box<!fir.array<2x2xi32>>, !fir.ref<!fir.array<2x2xi32>>)
end subroutine
