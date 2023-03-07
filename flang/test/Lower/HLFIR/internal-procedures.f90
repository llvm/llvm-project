! Test captured variables instantiation inside internal procedures
! when lowering to HLFIR.
! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s
subroutine test_explicit_shape_array(x, n)
  integer(8) :: n
  real :: x(n)
contains
subroutine internal
  call takes_array(x)
end subroutine
end subroutine
! CHECK-LABEL: func.func @_QFtest_explicit_shape_arrayPinternal(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.box<!fir.array<?xf32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.box<!fir.array<?xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_5]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_6]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_7]]) {uniq_name = "_QFtest_explicit_shape_arrayEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)

subroutine test_assumed_shape(x)
  real :: x(:)
contains
subroutine internal
  call takes_array(x)
end subroutine
end subroutine
! CHECK-LABEL: func.func @_QFtest_assumed_shapePinternal(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<tuple<!fir.box<!fir.array<?xf32>>>> {fir.host_assoc}) attributes {fir.internal_proc} {
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.box<!fir.array<?xf32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_4]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_6:.*]] = fir.shift %[[VAL_5]]#0 : (index) -> !fir.shift<1>
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_6]]) {uniq_name = "_QFtest_assumed_shapeEx"} : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)

subroutine test_scalar_char(c)
 character(*) :: c
contains
subroutine internal()
  call bar(c)
end subroutine
end subroutine
! CHECK-LABEL:   func.func @_QFtest_scalar_charPinternal(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<tuple<!fir.boxchar<1>>> {fir.host_assoc}) attributes {fir.internal_proc} {
! CHECK:  %[[VAL_1:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_1]] : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.boxchar<1>>
! CHECK:  %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_3]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]]#0 typeparams %[[VAL_4]]#1 {uniq_name = "_QFtest_scalar_charEc"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  fir.call @_QPbar(%[[VAL_5]]#0) {{.*}}: (!fir.boxchar<1>) -> ()
