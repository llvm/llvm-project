! Test how transformational intrinsic function references are lowered

! RUN: bbc -emit-fir %s -o - | FileCheck %s

! The exact intrinsic being tested does not really matter, what is
! tested here is that transformational intrinsics are lowered correctly
! regardless of the context they appear into.



module test2
interface
  subroutine takes_array_desc(l)
    logical(1) :: l(:)
  end subroutine
end interface

contains

! CHECK-LABEL: func @_QMtest2Pin_io(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>) {
subroutine in_io(x)
  logical(1) :: x(:, :)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[res_arg:.*]] = fir.convert %[[res_desc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[x_arg:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim(%[[res_arg]], %[[x_arg]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[io_embox:.*]] = fir.embox %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: %[[io_embox_cast:.*]] = fir.convert %[[io_embox]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}ioOutputDescriptor({{.*}}, %[[io_embox_cast]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
  print *, all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_call(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>) {
subroutine in_call(x)
  implicit none
  logical(1) :: x(:, :)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[res_arg:.*]] = fir.convert %[[res_desc]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[x_arg:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x?x!fir.logical<1>>>) -> !fir.box<none>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim(%[[res_arg]], %[[x_arg]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[call_embox:.*]] = fir.embox %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<1>>>
  ! CHECK: fir.call @_QPtakes_array_desc(%[[call_embox]]) : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> ()
  call takes_array_desc(all(x, 1))
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_implicit_call(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>) {
subroutine in_implicit_call(x)
  logical(1) :: x(:, :)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim
  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK: %[[res_addr_cast:.*]] = fir.convert %[[res_addr]] : (!fir.heap<!fir.array<?x!fir.logical<1>>>) -> !fir.ref<!fir.array<?x!fir.logical<1>>>
  ! CHECK: fir.call @_QPtakes_implicit_array(%[[res_addr_cast]]) : (!fir.ref<!fir.array<?x!fir.logical<1>>>) -> ()
  call takes_implicit_array(all(x, 1))
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_assignment(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>)
subroutine in_assignment(x, y)
  logical(1) :: x(:, :), y(:)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK: %[[y_load:.*]] = fir.array_load %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim

  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[res_load:.*]] = fir.array_load %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.array<?x!fir.logical<1>>

  ! CHECK: %[[assign:.*]] = fir.do_loop %[[idx:.*]] = %{{.*}} to {{.*}} {
    ! CHECK: %[[res_elt:.*]] = fir.array_fetch %[[res_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK: fir.array_update %{{.*}} %[[res_elt]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, !fir.logical<1>, index) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[y_load]], %[[assign]] to %[[arg1]] : !fir.box<!fir.array<?x!fir.logical<1>>>
  y = all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

! CHECK-LABEL: func @_QMtest2Pin_elem_expr(
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<1>>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>,
! CHECK-SAME: %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<1>>>)
subroutine in_elem_expr(x, y, z)
  logical(1) :: x(:, :), y(:), z(:)
  ! CHECK: %[[res_desc:.]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>
  ! CHECK-DAG: %[[y_load:.*]] = fir.array_load %[[arg1]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK-DAG: %[[z_load:.*]] = fir.array_load %[[arg2]] : (!fir.box<!fir.array<?x!fir.logical<1>>>) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: fir.call @_Fortran{{.*}}AllDim

  ! CHECK: %[[res_desc_load:.*]] = fir.load %[[res_desc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>>
  ! CHECK-DAG: %[[dims:.*]]:3 = fir.box_dims %[[res_desc_load]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[res_addr:.*]] = fir.box_addr %[[res_desc_load]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<1>>>>) -> !fir.heap<!fir.array<?x!fir.logical<1>>>
  ! CHECK-DAG: %[[res_shape:.*]] = fir.shape_shift %[[dims]]#0, %[[dims]]#1 : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[res_load:.*]] = fir.array_load %[[res_addr]](%[[res_shape]]) : (!fir.heap<!fir.array<?x!fir.logical<1>>>, !fir.shapeshift<1>) -> !fir.array<?x!fir.logical<1>>

  ! CHECK: %[[elem_expr:.*]] = fir.do_loop %[[idx:.*]] = %{{.*}} to {{.*}} {
    ! CHECK-DAG: %[[y_elt:.*]] = fir.array_fetch %[[y_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK-DAG: %[[res_elt:.*]] = fir.array_fetch %[[res_load]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, index) -> !fir.logical<1>
    ! CHECK-DAG: %[[y_elt_i1:.*]] = fir.convert %[[y_elt]] : (!fir.logical<1>) -> i1
    ! CHECK-DAG: %[[res_elt_i1:.*]] = fir.convert %[[res_elt]] : (!fir.logical<1>) -> i1
    ! CHECK: %[[neqv_i1:.*]] = cmpi ne, %[[y_elt_i1]], %[[res_elt_i1]] : i1
    ! CHECK: %[[neqv:.*]] = fir.convert %[[neqv_i1]] : (i1) -> !fir.logical<1>
    ! CHECK: fir.array_update %{{.*}} %[[neqv]], %[[idx]] : (!fir.array<?x!fir.logical<1>>, !fir.logical<1>, index) -> !fir.array<?x!fir.logical<1>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[z_load]], %[[elem_expr]] to %[[arg2]] : !fir.box<!fir.array<?x!fir.logical<1>>>
  z = y .neqv. all(x, 1)
  ! CHECK: fir.freemem %[[res_addr]] : !fir.heap<!fir.array<?x!fir.logical<1>>>
end subroutine

end module
