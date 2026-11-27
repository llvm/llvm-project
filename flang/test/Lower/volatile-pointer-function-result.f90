! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s

! A pointer or allocatable function result with the VOLATILE attribute must be
! cast to the non-volatile function result type when returned.

function func() result(res)
  integer, pointer :: res
  volatile :: res
  allocate(res)
  res = 7
end function

! CHECK-LABEL: func.func @_QPfunc() -> !fir.box<!fir.ptr<i32>> {
! CHECK:         %[[CAST:.*]] = fir.volatile_cast %{{.*}} : (!fir.box<!fir.ptr<i32>, volatile>) -> !fir.box<!fir.ptr<i32>>
! CHECK-NEXT:    return %[[CAST]] : !fir.box<!fir.ptr<i32>>

function func_alloc_array() result(res)
  integer, allocatable :: res(:)
  volatile :: res
  allocate(res(3))
  res = 1
end function

! CHECK-LABEL: func.func @_QPfunc_alloc_array() -> !fir.box<!fir.heap<!fir.array<?xi32>>> {
! CHECK:         %[[SHIFT:.*]] = fir.shift %{{.*}} : (index) -> !fir.shift<1>
! CHECK-NEXT:    %[[REBOX:.*]] = fir.rebox %{{.*}}(%[[SHIFT]]) : (!fir.box<!fir.heap<!fir.array<?xi32>>, volatile>, !fir.shift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>, volatile>
! CHECK-NEXT:    %[[CAST:.*]] = fir.volatile_cast %[[REBOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>, volatile>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK-NEXT:    return %[[CAST]] : !fir.box<!fir.heap<!fir.array<?xi32>>>

function func_ptr_array() result(res)
  integer, pointer :: res(:)
  volatile :: res
  allocate(res(3))
  res = 1
end function

! CHECK-LABEL: func.func @_QPfunc_ptr_array() -> !fir.box<!fir.ptr<!fir.array<?xi32>>> {
! CHECK:         %[[CAST:.*]] = fir.volatile_cast %{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK-NEXT:    return %[[CAST]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>

function func_class_alloc() result(res)
  class(*), allocatable :: res
  volatile :: res
  allocate(res, source=0)
end function

! CHECK-LABEL: func.func @_QPfunc_class_alloc() -> !fir.class<!fir.heap<none>> {
! CHECK:         %[[CAST:.*]] = fir.volatile_cast %{{.*}} : (!fir.class<!fir.heap<none>, volatile>) -> !fir.class<!fir.heap<none>>
! CHECK-NEXT:    return %[[CAST]] : !fir.class<!fir.heap<none>>

function func_char_ptr() result(res)
  character(len=4), pointer :: res
  volatile :: res
  allocate(res)
  res = "abcd"
end function

! CHECK-LABEL: func.func @_QPfunc_char_ptr() -> !fir.box<!fir.ptr<!fir.char<1,4>>> {
! CHECK:         fir.volatile_cast %{{.*}} : (!fir.box<!fir.ptr<!fir.char<1,4>>, volatile>) -> !fir.box<!fir.ptr<!fir.char<1,4>>>
! CHECK-NEXT:    return

function func_dt_alloc() result(res)
  type t
    integer :: i
  end type
  type(t), allocatable :: res
  volatile :: res
  allocate(res)
  res%i = 42
end function

! CHECK-LABEL: func.func @_QPfunc_dt_alloc() -> !fir.box<!fir.heap<!fir.type<
! CHECK:         fir.volatile_cast %{{.*}} : (!fir.box<!fir.heap<!fir.type<{{.*}}>, volatile>) -> !fir.box<!fir.heap<!fir.type<
! CHECK-NEXT:    return
