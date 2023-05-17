! Test lowering of component reference to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s
module comp_ref
type t1
  integer :: scalar_i
  real :: scalar_x
end type

type t2
  integer :: scalar_i2
  type(t1) :: scalar_t1
end type

type t_char
  integer :: scalar_i
  character(5) :: scalar_char
end type

type t_array
  integer :: scalar_i
  real :: array_comp(10,20)
end type

type t_array_lbs
  integer :: scalar_i
  real :: array_comp_lbs(2:11,3:22)
end type

type t_array_char
  integer :: scalar_i
  character(5) :: array_char_comp(10,20)
end type
end module

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                            Test scalar bases                                 !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine test_scalar(a)
  use comp_ref
  type(t1) :: a
  call use_real_scalar(a%scalar_x)
! CHECK-LABEL: func.func @_QPtest_scalar(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"scalar_x"}   : (!fir.ref<!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>) -> !fir.ref<f32>
end subroutine

subroutine test_scalar_char(a)
  use comp_ref
  type(t_char) :: a
  call use_char_scalar(a%scalar_char)
! CHECK-LABEL: func.func @_QPtest_scalar_char(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_3:.*]] = hlfir.designate %[[VAL_1]]#0{"scalar_char"}   typeparams %[[VAL_2]] : (!fir.ref<!fir.type<_QMcomp_refTt_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>, index) -> !fir.ref<!fir.char<1,5>>
end subroutine

subroutine test_scalar_char_substring(a)
  use comp_ref
  type(t_char) :: a
  call use_char_scalar(a%scalar_char(3:))
! CHECK-LABEL: func.func @_QPtest_scalar_char_substring(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_4:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_1]]#0{"scalar_char"}  substr %[[VAL_2]], %[[VAL_3]]  typeparams %[[VAL_4]] : (!fir.ref<!fir.type<_QMcomp_refTt_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>, index, index, index) -> !fir.ref<!fir.char<1,3>>
end subroutine

subroutine test_array_comp_1(a)
  use comp_ref
  type(t_array) :: a
  call use_real_array(a%array_comp)
! CHECK-LABEL: func.func @_QPtest_array_comp_1(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_5:.*]] = hlfir.designate %[[VAL_1]]#0{"array_comp"}   shape %[[VAL_4]] : (!fir.ref<!fir.type<_QMcomp_refTt_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.shape<2>) -> !fir.ref<!fir.array<10x20xf32>>
end subroutine

subroutine test_array_comp_slice(a)
  use comp_ref
  type(t_array) :: a
  ! Contiguous
  call use_real_array(a%array_comp(:, 4:20:1))
! CHECK-LABEL: func.func @_QPtest_array_comp_slice(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_8:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_9:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 17 : index
! CHECK:  %[[VAL_12:.*]] = fir.shape %[[VAL_7]], %[[VAL_11]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_13:.*]] = hlfir.designate %[[VAL_1]]#0{"array_comp"} <%[[VAL_4]]> (%[[VAL_5]]:%[[VAL_2]]:%[[VAL_6]], %[[VAL_8]]:%[[VAL_9]]:%[[VAL_10]])  shape %[[VAL_12]] : (!fir.ref<!fir.type<_QMcomp_refTt_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.shape<2>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<10x17xf32>>
end subroutine

subroutine test_array_comp_non_contiguous_slice(a)
  use comp_ref
  type(t_array) :: a
  ! Not contiguous
  print *, a%array_comp(1:6:1, 4:20:1)
! CHECK-LABEL: func.func @_QPtest_array_comp_non_contiguous_slice(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_8:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 6 : index
! CHECK:  %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_13:.*]] = arith.constant 6 : index
! CHECK:  %[[VAL_14:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_15:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_17:.*]] = arith.constant 17 : index
! CHECK:  %[[VAL_18:.*]] = fir.shape %[[VAL_13]], %[[VAL_17]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_19:.*]] = hlfir.designate %[[VAL_1]]#0{"array_comp"} <%[[VAL_9]]> (%[[VAL_10]]:%[[VAL_11]]:%[[VAL_12]], %[[VAL_14]]:%[[VAL_15]]:%[[VAL_16]])  shape %[[VAL_18]] : (!fir.ref<!fir.type<_QMcomp_refTt_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>, !fir.shape<2>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<6x17xf32>>
end subroutine

subroutine test_array_lbs_comp_lbs_1(a)
  use comp_ref
  type(t_array_lbs) :: a
  call use_real_array(a%array_comp_lbs)
! CHECK-LABEL: func.func @_QPtest_array_lbs_comp_lbs_1(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_6:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_2]], %[[VAL_5]], %[[VAL_3]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:  %[[VAL_7:.*]] = hlfir.designate %[[VAL_1]]#0{"array_comp_lbs"}   shape %[[VAL_6]] : (!fir.ref<!fir.type<_QMcomp_refTt_array_lbs{scalar_i:i32,array_comp_lbs:!fir.array<10x20xf32>}>>, !fir.shapeshift<2>) -> !fir.box<!fir.array<10x20xf32>>
end subroutine

subroutine test_array_lbs_comp_lbs_slice(a)
  use comp_ref
  type(t_array_lbs) :: a
  ! Contiguous
  call use_real_array(a%array_comp_lbs(:, 4:20:1))
! CHECK-LABEL: func.func @_QPtest_array_lbs_comp_lbs_slice(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_5:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_6:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_2]], %[[VAL_5]], %[[VAL_3]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_8:.*]] = arith.addi %[[VAL_4]], %[[VAL_2]] : index
! CHECK:  %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_7]] : index
! CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_12:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_13:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_15:.*]] = arith.constant 17 : index
! CHECK:  %[[VAL_16:.*]] = fir.shape %[[VAL_11]], %[[VAL_15]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_17:.*]] = hlfir.designate %[[VAL_1]]#0{"array_comp_lbs"} <%[[VAL_6]]> (%[[VAL_4]]:%[[VAL_9]]:%[[VAL_10]], %[[VAL_12]]:%[[VAL_13]]:%[[VAL_14]])  shape %[[VAL_16]] : (!fir.ref<!fir.type<_QMcomp_refTt_array_lbs{scalar_i:i32,array_comp_lbs:!fir.array<10x20xf32>}>>, !fir.shapeshift<2>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<10x17xf32>>
end subroutine

subroutine test_array_char_comp_1(a)
  use comp_ref
  type(t_array_char) :: a
  call use_array_char(a%array_char_comp)
! CHECK-LABEL: func.func @_QPtest_array_char_comp_1(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_5:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_6:.*]] = hlfir.designate %[[VAL_1]]#0{"array_char_comp"}   shape %[[VAL_4]] typeparams %[[VAL_5]] : (!fir.ref<!fir.type<_QMcomp_refTt_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>, !fir.shape<2>, index) -> !fir.ref<!fir.array<10x20x!fir.char<1,5>>>
end subroutine

subroutine test_array_char_comp_slice(a)
  use comp_ref
  type(t_array_char) :: a
  ! Contiguous
  call use_array_char(a%array_char_comp(:, 4:20:1))
! CHECK-LABEL: func.func @_QPtest_array_char_comp_slice(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_5:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_8:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_9:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_10:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_12:.*]] = arith.constant 17 : index
! CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_8]], %[[VAL_12]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_14:.*]] = hlfir.designate %[[VAL_1]]#0{"array_char_comp"} <%[[VAL_4]]> (%[[VAL_6]]:%[[VAL_2]]:%[[VAL_7]], %[[VAL_9]]:%[[VAL_10]]:%[[VAL_11]])  shape %[[VAL_13]] typeparams %[[VAL_5]] : (!fir.ref<!fir.type<_QMcomp_refTt_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>, !fir.shape<2>, index, index, index, index, index, index, !fir.shape<2>, index) -> !fir.ref<!fir.array<10x17x!fir.char<1,5>>>
end subroutine

subroutine test_array_char_comp_non_contiguous_slice(a)
  use comp_ref
  type(t_array_char) :: a
  ! Not contiguous
  print *, a%array_char_comp(1:10:1,1:20:1)(2:4)
! CHECK-LABEL: func.func @_QPtest_array_char_comp_non_contiguous_slice(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_7:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_8:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_7]], %[[VAL_8]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_13:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_15:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_17:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_18:.*]] = fir.shape %[[VAL_13]], %[[VAL_17]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_19:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_20:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_21:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_22:.*]] = hlfir.designate %[[VAL_1]]#0{"array_char_comp"} <%[[VAL_9]]> (%[[VAL_10]]:%[[VAL_11]]:%[[VAL_12]], %[[VAL_14]]:%[[VAL_15]]:%[[VAL_16]]) substr %[[VAL_19]], %[[VAL_20]]  shape %[[VAL_18]] typeparams %[[VAL_21]] : (!fir.ref<!fir.type<_QMcomp_refTt_array_char{scalar_i:i32,array_char_comp:!fir.array<10x20x!fir.char<1,5>>}>>, !fir.shape<2>, index, index, index, index, index, index, index, index, !fir.shape<2>, index) -> !fir.box<!fir.array<10x20x!fir.char<1,3>>>
end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                            Test array bases                                  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine test_array(a)
  use comp_ref
  type(t1) :: a(:)
  print *, a%scalar_x
! CHECK-LABEL: func.func @_QPtest_array(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_1]]#0, %[[VAL_7]] : (!fir.box<!fir.array<?x!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_8]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_10:.*]] = hlfir.designate %[[VAL_1]]#0{"scalar_x"}   shape %[[VAL_9]] : (!fir.box<!fir.array<?x!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
end subroutine

subroutine test_array_char(a, n)
  use comp_ref
  integer(8) :: n
  type(t_char) :: a(n)
  print *, a%scalar_char
! CHECK-LABEL: func.func @_QPtest_array_char(
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_8:[a-z0-9]*]])  {{.*}}Ea
! CHECK:  %[[VAL_15:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_16:.*]] = hlfir.designate %[[VAL_9]]#0{"scalar_char"}   shape %[[VAL_8]] typeparams %[[VAL_15]] : (!fir.box<!fir.array<?x!fir.type<_QMcomp_refTt_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,5>>>
end subroutine

subroutine test_array_char_substring(a)
  use comp_ref
  type(t_char) :: a(100)
  print *, a%scalar_char(3:)
! CHECK-LABEL: func.func @_QPtest_array_char_substring(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_2:[a-z0-9]*]])  {{.*}}Ea
! CHECK:  %[[VAL_9:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_10:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_12:.*]] = hlfir.designate %[[VAL_3]]#0{"scalar_char"}  substr %[[VAL_9]], %[[VAL_10]]  shape %[[VAL_2]] typeparams %[[VAL_11]] : (!fir.ref<!fir.array<100x!fir.type<_QMcomp_refTt_char{scalar_i:i32,scalar_char:!fir.char<1,5>}>>>, index, index, !fir.shape<1>, index) -> !fir.box<!fir.array<100x!fir.char<1,3>>>
end subroutine

subroutine test_array_array_comp_1(a)
  use comp_ref
  type(t_array) :: a(100)
  print *, a%array_comp(4,5)
! CHECK-LABEL: func.func @_QPtest_array_array_comp_1(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_2:[a-z0-9]*]])  {{.*}}Ea
! CHECK:  %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_10:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_11:.*]] = fir.shape %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_12:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_13:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_14:.*]] = hlfir.designate %[[VAL_3]]#0{"array_comp"} <%[[VAL_11]]> (%[[VAL_12]], %[[VAL_13]])  shape %[[VAL_2]] : (!fir.ref<!fir.array<100x!fir.type<_QMcomp_refTt_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shape<2>, index, index, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Test several part ref (produces chain of hlfir.designate)                    !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine test_scalar_chain(a)
  use comp_ref
  type(t2) :: a
  call use_real_scalar(a%scalar_t1%scalar_x)
! CHECK-LABEL: func.func @_QPtest_scalar_chain(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ea
! CHECK:  %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"scalar_t1"}   : (!fir.ref<!fir.type<_QMcomp_refTt2{scalar_i2:i32,scalar_t1:!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>}>>) -> !fir.ref<!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>
! CHECK:  %[[VAL_3:.*]] = hlfir.designate %[[VAL_2]]{"scalar_x"}   : (!fir.ref<!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>) -> !fir.ref<f32>
end subroutine

subroutine test_array_scalar_chain(a)
  use comp_ref
  type(t2) :: a(100)
  print *, a%scalar_t1%scalar_x
! CHECK-LABEL: func.func @_QPtest_array_scalar_chain(
! CHECK:  %[[VAL_1:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_2:[a-z0-9]*]])  {{.*}}Ea
! CHECK:  %[[VAL_9:.*]] = hlfir.designate %[[VAL_3]]#0{"scalar_t1"}   shape %[[VAL_2]] : (!fir.ref<!fir.array<100x!fir.type<_QMcomp_refTt2{scalar_i2:i32,scalar_t1:!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<100x!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>>
! CHECK:  %[[VAL_10:.*]] = hlfir.designate %[[VAL_9]]{"scalar_x"}   shape %[[VAL_2]] : (!fir.box<!fir.array<100x!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
end subroutine

subroutine test_scalar_chain_2(a)
  use comp_ref
  type(t1) :: a(50)
  print *, a(10)%scalar_x
! CHECK-LABEL: func.func @_QPtest_scalar_chain_2(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_2:[a-z0-9]*]])  {{.*}}Ea
! CHECK:  %[[VAL_9:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_10:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<50x!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>>, index) -> !fir.ref<!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>
! CHECK:  %[[VAL_11:.*]] = hlfir.designate %[[VAL_10]]{"scalar_x"}   : (!fir.ref<!fir.type<_QMcomp_refTt1{scalar_i:i32,scalar_x:f32}>>) -> !fir.ref<f32>
end subroutine

subroutine test_array_ref_chain(a)
  use comp_ref
  type(t_array) :: a(100)
  print *, a(1:50:5)%array_comp(4,5)
! CHECK-LABEL: func.func @_QPtest_array_ref_chain(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_2:[a-z0-9]*]])  {{.*}}Ea
! CHECK:  %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_10:.*]] = arith.constant 50 : index
! CHECK:  %[[VAL_11:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_12:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_14:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_9]]:%[[VAL_10]]:%[[VAL_11]])  shape %[[VAL_13]] : (!fir.ref<!fir.array<100x!fir.type<_QMcomp_refTt_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<_QMcomp_refTt_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>
! CHECK:  %[[VAL_15:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_16:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_17:.*]] = fir.shape %[[VAL_15]], %[[VAL_16]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_18:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_19:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_20:.*]] = hlfir.designate %[[VAL_14]]{"array_comp"} <%[[VAL_17]]> (%[[VAL_18]], %[[VAL_19]])  shape %[[VAL_13]] : (!fir.box<!fir.array<10x!fir.type<_QMcomp_refTt_array{scalar_i:i32,array_comp:!fir.array<10x20xf32>}>>>, !fir.shape<2>, index, index, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
end subroutine
