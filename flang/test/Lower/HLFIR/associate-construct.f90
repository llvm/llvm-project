! Test lowering of associate construct to HLFIR
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine associate_expr(x)
  integer :: x(:)
  associate(y => x + 42)
    print *, y
  end associate
end subroutine
! CHECK-LABEL: func.func @_QPassociate_expr(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_1]]#0, %[[VAL_3]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_6:.*]] = hlfir.elemental {{.*}}
! CHECK:  %[[VAL_11:.*]]:3 = hlfir.associate %[[VAL_6]]{{.*}}
! CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_4]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_11]]#1(%[[VAL_13]]) {uniq_name = "_QFassociate_exprEy"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
! CHECK:  fir.call @_FortranAioEndIoStatement
! CHECK:  hlfir.end_associate %[[VAL_11]]#1, %[[VAL_11]]#2 : !fir.ref<!fir.array<?xi32>>, i1

subroutine associate_var(x)
  integer :: x
  associate(y => x)
    print *, y
  end associate
end subroutine
! CHECK-LABEL: func.func @_QPassociate_var(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#1 {uniq_name = "_QFassociate_varEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  fir.call @_FortranAioEndIoStatement
! CHECK-NEXT:  return

subroutine associate_pointer(x)
  integer, pointer, contiguous :: x(:)
  ! Check that "y" has the target attribute.
  associate(y => x)
    print *, y
  end associate
end subroutine
! CHECK-LABEL: func.func @_QPassociate_pointer(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_6]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFassociate_pointerEy"} : (!fir.ptr<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ptr<!fir.array<?xi32>>)
! CHECK:  fir.call @_FortranAioEndIoStatement
! CHECK-NEXT:  return

subroutine associate_allocatable(x)
  integer, allocatable :: x(:)
  associate(y => x)
    print *, y
  end associate
end subroutine
! CHECK-LABEL: func.func @_QPassociate_allocatable(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_6:.*]] = fir.shape_shift %[[VAL_5]]#0, %[[VAL_5]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_6]]) {uniq_name = "_QFassociate_allocatableEy"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:  fir.call @_FortranAioEndIoStatement
! CHECK-NEXT:  return

subroutine associate_optional(x)
  integer, optional :: x(:)
  ! Check that "y" is not given the optional attribute: x must be present as per
  ! Fortran 2018 11.1.3.2 point 4.
  associate(y => x)
    print *, y
  end associate
end subroutine
! CHECK-LABEL: func.func @_QPassociate_optional(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#1 {uniq_name = "_QFassociate_optionalEy"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:  fir.call @_FortranAioEndIoStatement
! CHECK-NEXT:  return

subroutine associate_pointer_section(x)
  integer , pointer, contiguous :: x(:)
  associate (y => x(1:20:1))
    print *, y
  end associate
end subroutine
! CHECK-LABEL: func.func @_QPassociate_pointer_section(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex"
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_6:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_8:.*]] = hlfir.designate %[[VAL_2]]{{.*}}
! CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFassociate_pointer_sectionEy"} : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<20xi32>>, !fir.ref<!fir.array<20xi32>>)
! CHECK:  fir.call @_FortranAioEndIoStatement
! CHECK-NEXT:  return
