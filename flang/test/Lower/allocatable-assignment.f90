! Test allocatable assignments
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module alloc_assign
  type t
    integer :: i
  end type
contains

! -----------------------------------------------------------------------------
!            Test simple scalar RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_simple_scalar(
! CHECK-SAME: %[[box:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>) {
subroutine test_simple_scalar(x)
  real, allocatable  :: x
  ! CHECK: %[[cst:.*]] = constant 4.200000e+01 : f32
  ! CHECK: %[[boxLoad:.*]] = fir.load %[[box]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr:.*]] = fir.box_addr %[[boxLoad]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[addrCast:.*]] = fir.convert %[[addr]] : (!fir.heap<f32>) -> i64
  ! CHECK: %[[isAlloc:.*]] = cmpi ne, %[[addrCast]], %c0{{.*}} : i64
  ! CHECK: fir.if %[[isAlloc]] {
  ! CHECK:   fir.if %false{{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[alloc:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[embox:.*]] = fir.embox %[[alloc]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  ! CHECK:   fir.store %[[embox]] to %[[box]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: }
  ! CHECK: %[[boxLoad2:.*]] = fir.load %[[box]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr2:.*]] = fir.box_addr %[[boxLoad2]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: fir.store %[[cst]] to %[[addr2]] : !fir.heap<f32>
  x = 42.
end subroutine

subroutine test_simple_local_scalar()
  real, allocatable  :: x
  ! CHECK: %[[x:.*]] = fir.alloca !fir.heap<f32> {uniq_name = "_QMalloc_assignFtest_simple_local_scalarEx.addr"}
  ! CHECK: %[[cst:.*]] = constant 4.200000e+01 : f32
  ! CHECK: %[[xAddr:.*]] = fir.load %[[x]] : !fir.ref<!fir.heap<f32>>
  ! CHECK: %[[xCast:.*]] = fir.convert %[[xAddr]] : (!fir.heap<f32>) -> i64
  ! CHECK: %[[isAlloc:.*]] = cmpi ne, %[[xCast]], %c0{{.*}} : i64
  ! CHECK: fir.if %[[isAlloc]] {
  ! CHECK:   fir.if %false{{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[alloc:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
  ! CHECK:   fir.store %[[alloc]] to %[[x]] : !fir.ref<!fir.heap<f32>>
  ! CHECK: }
  ! CHECK: %[[xAddr2:.*]] = fir.load %[[x]] : !fir.ref<!fir.heap<f32>>
  ! CHECK: fir.store %[[cst]] to %[[xAddr2]] : !fir.heap<f32>
  x = 42.
  print *, x
end subroutine

! -----------------------------------------------------------------------------
!            Test character scalar RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_deferred_char_scalar(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) {
subroutine test_deferred_char_scalar(x)
  character(:), allocatable  :: x
  ! CHECK: %[[boxLoad:.*]] = fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK: %[[xAddr:.]] = fir.box_addr %[[boxLoad]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   %[[xLen:.*]] = fir.box_elesize %[[boxLoad]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
  ! CHECK:   %[[cmpLen:.*]] = cmpi ne, %[[xLen]], %c12{{.*}} : index
  ! CHECK:   %[[realloc:.*]] = select %[[cmpLen]], %[[cmpLen]], %false{{.*}} : i1
  ! CHECK:   fir.if %[[realloc]] {
  ! CHECK:     fir.freemem %[[xAddr]] : !fir.heap<!fir.char<1,?>>
  ! CHECK:     %[[newAddr:.*]] = fir.allocmem !fir.char<1,?>(%c12{{.*}} : index) {uniq_name = ".auto.alloc"}
  ! CHECK:     %[[box:.*]] = fir.embox %[[newAddr]] typeparams %c12{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK:     fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.char<1,?>(%c12{{.*}} : index) {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]] typeparams %c12{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK: }
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_cst_char_scalar(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>) {
subroutine test_cst_char_scalar(x)
  character(10), allocatable  :: x
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.if %false{{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.char<1,10> {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]] : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
  ! CHECK: }
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_dyn_char_scalar(
! CHECK: %[[x:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, %[[nAddr:.*]]: !fir.ref<i32>) {
subroutine test_dyn_char_scalar(x, n)
  integer :: n
  character(n), allocatable  :: x
  ! CHECK: %[[n:.*]] = fir.load %[[nAddr]] : !fir.ref<i32>
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.if %false_0 {
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[nCast:.*]] = fir.convert %[[n]] : (i32) -> index
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.char<1,?>(%[[nCast]] : index) {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]] typeparams %[[nCast]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK: }
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_derived_scalar(
! CHECK-SAME: %[[box:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>, %{{.*}}) {
subroutine test_derived_scalar(x, s)
  type(t), allocatable  :: x
  type(t) :: s
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.if %false{{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.type<_QMalloc_assignTt{i:i32}> {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]] : (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) -> !fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>
  ! CHECK: }
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>
  x = s
end subroutine

! -----------------------------------------------------------------------------
!            Test numeric/logical array RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_from_cst_shape_array(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %{{.*}}) {
subroutine test_from_cst_shape_array(x, y)
  real, allocatable  :: x(:, :)
  real :: y(2, 3)
  ! CHECK: %[[boxLoad:.*]] = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   %[[dim1:.*]]:3 = fir.box_dims %[[boxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK:   %[[dim2:.*]]:3 = fir.box_dims %[[boxLoad]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK:   %[[cmp1:.*]] = cmpi ne, %[[dim1]]#1, %c2{{.*}} : index
  ! CHECK:   %[[mustRealloc1:.*]] = select %[[cmp1]], %[[cmp1]], %false : i1
  ! CHECK:   %[[cmp2:.*]] = cmpi ne, %[[dim2]]#1, %c3{{.*}} : index
  ! CHECK:   %[[mustRealloc2:.*]] = select %[[cmp2]], %[[cmp2]], %[[mustRealloc1]] : i1
  ! CHECK:   fir.if %[[mustRealloc2]] {
  ! CHECK:     fir.freemem %{{.*}} : !fir.heap<!fir.array<?x?xf32>>
  ! CHECK:     %[[newAddr:.*]] = fir.allocmem !fir.array<?x?xf32>, %c2{{.*}}, %c3{{.*}} {uniq_name = ".auto.alloc"}
  ! CHECK:     %[[shape:.*]] = fir.shape %c2{{.*}}, %c3{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK:     %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK:     fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.array<?x?xf32>, %c2{{.*}}, %c3{{.*}} {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[shape:.*]] = fir.shape %c2{{.*}}, %c3{{.*}} : (index, index) -> !fir.shape<2>
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: }
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  x = y
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_from_dyn_shape_array(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %[[y:.*]]: !fir.box<!fir.array<?x?xf32>>) {
subroutine test_from_dyn_shape_array(x, y)
  real, allocatable  :: x(:, :)
  ! CHECK: %[[ydim1:.*]]:3 = fir.box_dims %[[y]], %c0{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK: %[[ydim2:.*]]:3 = fir.box_dims %[[y]], %c1{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  real :: y(:, :)
  ! CHECK: %[[boxLoad:.*]] = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   %[[dim1:.*]]:3 = fir.box_dims %[[boxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK:   %[[dim2:.*]]:3 = fir.box_dims %[[boxLoad]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK:   %[[cmp1:.*]] = cmpi ne, %[[dim1]]#1, %[[ydim1]]#1 : index
  ! CHECK:   %[[mustRealloc1:.*]] = select %[[cmp1]], %[[cmp1]], %false : i1
  ! CHECK:   %[[cmp2:.*]] = cmpi ne, %[[dim2]]#1, %[[ydim2]]#1 : index
  ! CHECK:   %[[mustRealloc2:.*]] = select %[[cmp2]], %[[cmp2]], %[[mustRealloc1]] : i1
  ! CHECK:   fir.if %[[mustRealloc2]] {
  ! CHECK:     fir.freemem %{{.*}} : !fir.heap<!fir.array<?x?xf32>>
  ! CHECK:     %[[newAddr:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[ydim1]]#1, %[[ydim2]]#1 {uniq_name = ".auto.alloc"}
  ! CHECK:     %[[shape:.*]] = fir.shape %[[ydim1]]#1, %[[ydim2]]#1 : (index, index) -> !fir.shape<2>
  ! CHECK:     %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK:     fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[ydim1]]#1, %[[ydim2]]#1 {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[shape:.*]] = fir.shape %[[ydim1]]#1, %[[ydim2]]#1 : (index, index) -> !fir.shape<2>
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: }
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  x = y
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_with_lbounds(
! CHECK-SAME: %[[x:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>, %[[y:.*]]: !fir.box<!fir.array<?x?xf32>>) {
subroutine test_with_lbounds(x, y)
  real, allocatable  :: x(:, :)
  ! CHECK-DAG: %[[lb1:.*]] = fir.convert %c10{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[lb2:.*]] = fir.convert %c20{{.*}} : (i64) -> index
  ! CHECK-DAG: %[[ydim1:.*]]:3 = fir.box_dims %[[y]], %c0{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[ydim2:.*]]:3 = fir.box_dims %[[y]], %c1{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
  real :: y(10:, 20:)
  ! CHECK: %[[boxLoad:.*]] = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   %[[dim1:.*]]:3 = fir.box_dims %[[boxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK:   %[[dim2:.*]]:3 = fir.box_dims %[[boxLoad]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK:   %[[cmp1:.*]] = cmpi ne, %[[dim1]]#1, %[[ydim1]]#1 : index
  ! CHECK:   %[[mustRealloc1:.*]] = select %[[cmp1]], %[[cmp1]], %false : i1
  ! CHECK:   %[[cmp2:.*]] = cmpi ne, %[[dim2]]#1, %[[ydim2]]#1 : index
  ! CHECK:   %[[mustRealloc2:.*]] = select %[[cmp2]], %[[cmp2]], %[[mustRealloc1]] : i1
  ! CHECK:   fir.if %[[mustRealloc2]] {
  ! CHECK:     fir.freemem %{{.*}} : !fir.heap<!fir.array<?x?xf32>>
  ! CHECK:     %[[newAddr:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[ydim1]]#1, %[[ydim2]]#1 {uniq_name = ".auto.alloc"}
  ! CHECK:     %[[shape:.*]] = fir.shape_shift %[[lb1]], %[[ydim1]]#1, %[[lb2]], %[[ydim2]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
  ! CHECK:     %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK:     fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[ydim1]]#1, %[[ydim2]]#1 {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[shape:.*]] = fir.shape_shift %[[lb1]], %[[ydim1]]#1, %[[lb2]], %[[ydim2]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: }
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  x = y
end subroutine

subroutine test_runtime_shape(x)
  real, allocatable  :: x(:, :)
  interface
   function return_pointer()
     real, pointer :: return_pointer(:, :)
   end function
  end interface
  ! CHECK: %[[call:.*]] = fir.call @_QPreturn_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
  ! CHECK: fir.save_result %[[call]] to %[[resultStorage:.*]] : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  ! CHECK: %[[result:.*]] = fir.load %[[resultStorage]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  ! CHECK: fir.array_load
  ! CHECK-DAG: %[[ydim1:.*]]:3 = fir.box_dims %[[result]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK-DAG: %[[ydim2:.*]]:3 = fir.box_dims %[[result]], %c1{{.*}} : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)

  ! CHECK: %[[boxLoad:.*]] = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   %[[dim1:.*]]:3 = fir.box_dims %[[boxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK:   %[[dim2:.*]]:3 = fir.box_dims %[[boxLoad]], %c1{{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  ! CHECK:   %[[cmp1:.*]] = cmpi ne, %[[dim1]]#1, %[[ydim1]]#1 : index
  ! CHECK:   %[[mustRealloc1:.*]] = select %[[cmp1]], %[[cmp1]], %false : i1
  ! CHECK:   %[[cmp2:.*]] = cmpi ne, %[[dim2]]#1, %[[ydim2]]#1 : index
  ! CHECK:   %[[mustRealloc2:.*]] = select %[[cmp2]], %[[cmp2]], %[[mustRealloc1]] : i1
  ! CHECK:   fir.if %[[mustRealloc2]] {
  ! CHECK:     fir.freemem %{{.*}} : !fir.heap<!fir.array<?x?xf32>>
  ! CHECK:     %[[newAddr:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[ydim1]]#1, %[[ydim2]]#1 {uniq_name = ".auto.alloc"}
  ! CHECK:     %[[shape:.*]] = fir.shape %[[ydim1]]#1, %[[ydim2]]#1 : (index, index) -> !fir.shape<2>
  ! CHECK:     %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK:     fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[ydim1]]#1, %[[ydim2]]#1 {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[shape:.*]] = fir.shape %[[ydim1]]#1, %[[ydim2]]#1 : (index, index) -> !fir.shape<2>
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK: }

  ! CHECK-NOT: fir.call @_QPreturn_pointer()
  ! CHECK: fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
  ! CHECK-NOT: fir.call @_QPreturn_pointer()
  x = return_pointer()
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_scalar_rhs(
subroutine test_scalar_rhs(x, y)
  real, allocatable  :: x(:)
  real :: y
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.if %false {
  ! CHECK:   }
  ! CHECK: } else {
  ! TODO: runtime error if unallocated
  ! CHECK-NOT: allocmem
  ! CHECK: }
  x = y
end subroutine

! -----------------------------------------------------------------------------
!            Test character array RHS
! -----------------------------------------------------------------------------


! Hit TODO: gathering lhs length in array expression
!subroutine test_deferred_char_rhs_scalar(x)
!  character(:), allocatable  :: x(:)
!  x = "Hello world!"
!end subroutine

! CHECK: func @_QMalloc_assignPtest_cst_char_rhs_scalar(
subroutine test_cst_char_rhs_scalar(x)
  character(10), allocatable  :: x(:)
  x = "Hello world!"
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.if %false {
  ! CHECK:   }
  ! CHECK: } else {
  ! TODO: runtime error if unallocated
  ! CHECK-NOT: allocmem
  ! CHECK: }
end subroutine

! CHECK: func @_QMalloc_assignPtest_dyn_char_rhs_scalar(
subroutine test_dyn_char_rhs_scalar(x, n)
  integer :: n
  character(n), allocatable  :: x(:)
  x = "Hello world!"
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   fir.if %false {
  ! CHECK:   }
  ! CHECK: } else {
  ! TODO: runtime error if unallocated
  ! CHECK-NOT: allocmem
  ! CHECK: }
end subroutine

! Hit TODO: gathering lhs length in array expression
!subroutine test_deferred_char(x, c)
!  character(:), allocatable  :: x(:)
!  character(12) :: c(20)
!  x = "Hello world!"
!end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_cst_char(
! CHECK-SAME: %[[x]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>, %{{.*}}) {
subroutine test_cst_char(x, c)
  character(10), allocatable  :: x(:)
  character(12) :: c(20)
  ! CHECK: %[[boxLoad:.*]] = fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   %[[dim:.*]]:3 = fir.box_dims %[[boxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, index) -> (index, index, index)
  ! CHECK:   %[[cmp:.*]] = cmpi ne, %[[dim]]#1, %c20{{.*}} : index
  ! CHECK:   %[[mustRealloc:.*]] = select %[[cmp]], %[[cmp]], %false : i1
  ! CHECK:   fir.if %[[mustRealloc]] {
  ! CHECK:     fir.freemem %{{.*}} : !fir.heap<!fir.array<?x!fir.char<1,10>>>
  ! CHECK:     %[[newAddr:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %c20{{.*}} {uniq_name = ".auto.alloc"}
  ! CHECK:     %[[shape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK:     fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %c20{{.*}} {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[shape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  ! CHECK: }
  ! CHECK: %[[boxLoad:.*]] = fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  x = c
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_dyn_char(
! CHECK-SAME: %[[x]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, %[[nAddr:.*]]: !fir.ref<i32>, %{{.*}}) {
subroutine test_dyn_char(x, n, c)
  integer :: n
  character(n), allocatable  :: x(:)
  character(*) :: c(20)
  ! CHECK: %[[n:.*]] = fir.load %[[nAddr]] : !fir.ref<i32>
  ! CHECK: %[[boxLoad:.*]] = fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: fir.if %{{.*}} {
  ! CHECK:   %[[dim:.*]]:3 = fir.box_dims %[[boxLoad]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, index) -> (index, index, index)
  ! CHECK:   %[[cmp:.*]] = cmpi ne, %[[dim]]#1, %c20{{.*}} : index
  ! CHECK:   %[[mustRealloc:.*]] = select %[[cmp]], %[[cmp]], %false : i1
  ! CHECK:   fir.if %[[mustRealloc]] {
  ! CHECK:     fir.freemem %{{.*}} : !fir.heap<!fir.array<?x!fir.char<1,?>>>
  ! CHECK:     %[[nCast:.*]] = fir.convert %[[n]] : (i32) -> index
  ! CHECK:     %[[newAddr:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[nCast]] : index), %c20{{.*}} {uniq_name = ".auto.alloc"}
  ! CHECK:     %[[shape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:     %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) typeparams %[[nCast]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK:     fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK:   %[[nCast:.*]] = fir.convert %[[n]] : (i32) -> index
  ! CHECK:   %[[newAddr:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[nCast]] : index), %c20{{.*}} {uniq_name = ".auto.alloc"}
  ! CHECK:   %[[shape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK:   %[[box:.*]] = fir.embox %[[newAddr]](%[[shape]]) typeparams %[[nCast]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK:   fir.store %[[box]] to %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  ! CHECK: }
  ! CHECK: %[[boxLoad:.*]] = fir.load %[[x]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  x = c
end subroutine

end module

!  use alloc_assign
!  real :: y(2, 3) = reshape([1,2,3,4,5,6], [2,3])
!  real, allocatable :: x (:, :)
!  allocate(x(2,2))
!  call test_with_lbounds(x, y) 
!  print *, x(10, 20)
!  print *, x
!end
