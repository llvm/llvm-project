! Test lowering of structure constructors
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module m_struct_ctor
  implicit none
  type t_simple
    real :: x
  end type
  type t_char_scalar
    real :: x
    character(3) :: c
  end type
  type t_array
    real :: x
    integer :: i(5)
  end type
  type t_char_array
    real :: x
    character(3) :: c(5)
  end type
  type t_ptr
    real :: x
    integer, pointer :: p(:,:)
  end type
  type t_nested
    real :: x
    type(t_array) :: dt
  end type
contains
  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_simple(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>)
  subroutine test_simple(x)
    real :: x
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_simple{x:f32}> {uniq_name = {{.*}}}
    ! CHECK: %[[field:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_simple{x:f32}>
    ! CHECK: %[[xcoor:.*]] = fir.coordinate_of %[[tmp]], %[[field]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_simple{x:f32}>>, !fir.field) -> !fir.ref<f32>
    ! CHECK: %[[val:.*]] = fir.load %[[x]] : !fir.ref<f32>
    ! CHECK: fir.store %[[val]] to %[[xcoor]] : !fir.ref<f32>
    call print_simple(t_simple(x=x))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_char_scalar(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>)
  subroutine test_char_scalar(x)
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}> {uniq_name = {{.*}}}
    ! CHECK: %[[xfield:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>
    ! CHECK: %[[xcoor]] = fir.coordinate_of %[[tmp]], %[[xfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>>, !fir.field) -> !fir.ref<f32>
    ! CHECK: %[[val:.*]] = fir.load %[[x]] : !fir.ref<f32>
    ! CHECK: fir.store %[[val]] to %[[xcoor]] : !fir.ref<f32>

    ! CHECK: %[[cfield:.*]] = fir.field_index c, !fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>
    ! CHECK: %[[ccoor:.*]] = fir.coordinate_of %[[tmp]], %[[cfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>>, !fir.field) -> !fir.ref<!fir.char<1,3>>
    ! CHECK: %[[cst:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,3>>
    ! CHECK-DAG: %[[ccast:.*]] = fir.convert %[[ccoor]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
    ! CHECK-DAG: %[[cstcast:.*]] = fir.convert %[[cst]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
    ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[ccast]], %[[cstcast]], %{{.*}}, %{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
    real :: x
    call print_char_scalar(t_char_scalar(x=x, c="abc"))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_simple_array(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>, %[[j:.*]]: !fir.ref<!fir.array<5xi32>>)
  subroutine test_simple_array(x, j)
    real :: x
    integer :: j(5)
    call print_simple_array(t_array(x=x, i=2*j))
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}> {uniq_name = {{.*}}}
    ! CHECK: %[[xfield:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
    ! CHECK: %[[xcoor:.*]] = fir.coordinate_of %[[tmp]], %[[xfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<f32>
    ! CHECK: %[[val:.*]] = fir.load %[[x]] : !fir.ref<f32>
    ! CHECK: fir.store %[[val]] to %[[xcoor]] : !fir.ref<f32>

    ! CHECK: %[[ifield:.*]] = fir.field_index i, !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
    ! CHECK: %[[icoor:.*]] = fir.coordinate_of %[[tmp]], %[[ifield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<!fir.array<5xi32>>
    ! CHECK: %[[iload:.*]] = fir.array_load %[[icoor]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.array<5xi32>
    ! CHECK: %[[jload:.*]] = fir.array_load %[[j]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.array<5xi32>
    ! CHECK: %[[loop:.*]] = fir.do_loop %[[idx:.*]] = %c0{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%[[res:.*]] = %[[iload]]) -> (!fir.array<5xi32>) {
    ! CHECK:   %[[jval:.*]] = fir.array_fetch %[[jload]], %[[idx]] : (!fir.array<5xi32>, index) -> i32
    ! CHECK:   %[[ival:.*]] = muli %c2{{.*}}, %[[jval]] : i32
    ! CHECK:   %[[iupdate:.*]] = fir.array_update %[[res]], %[[ival]], %[[idx]] : (!fir.array<5xi32>, i32, index) -> !fir.array<5xi32>
    ! CHECK:   fir.result %[[iupdate]] : !fir.array<5xi32>
    ! CHECK: fir.array_merge_store %[[iload]], %[[loop]] to %[[icoor]] : !fir.ref<!fir.array<5xi32>>

  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_char_array(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>, %[[c1:.*]]: !fir.boxchar<1>) {
  subroutine test_char_array(x, c1)
    real :: x
    character(3) :: c1(5)
    ! CHECK: %1 = fir.alloca !fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}> {uniq_name = {{.*}}}
    ! CHECK: %[[unbox:.*]]:2 = fir.unboxchar %[[c1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK: %[[c1addr:.*]] = fir.convert %[[unbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<5x!fir.char<1,3>>>
    ! CHECK: fir.field_index x, !fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>

    ! CHECK: %[[cfield:.*]] = fir.field_index c, !fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>
    ! CHECK: %[[ccoor:.*]] = fir.coordinate_of %[[tmp]], %[[cfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>>, !fir.field) -> !fir.ref<!fir.array<5x!fir.char<1,3>>>
    ! CHECK: %[[cload:.*]] = fir.array_load %[[ccoor]](%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.array<5x!fir.char<1,3>>
    ! CHECK: %[[c1load:.*]] = fir.array_load %[[c1addr]](%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.array<5x!fir.char<1,3>>
    ! CHECK: %[[loop:.*]] = fir.do_loop %[[idx:.*]] = %c0{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%[[res:.*]] = %[[cload]]) -> (!fir.array<5x!fir.char<1,3>>) {
    ! CHECK:   %[[fetch:.*]] = fir.array_fetch %[[c1load]], %[[idx]] : (!fir.array<5x!fir.char<1,3>>, index) -> !fir.ref<!fir.char<1,3>>
    ! CHECK:   %[[update:.*]] = fir.array_update %[[res]], %[[fetch]], %[[idx]] : (!fir.array<5x!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>, index) -> !fir.array<5x!fir.char<1,3>>
    ! CHECK:   fir.result %[[update]] : !fir.array<5x!fir.char<1,3>>
    ! CHECK: fir.array_merge_store %[[cload]], %[[loop]] to %[[ccoor]] : !fir.ref<!fir.array<5x!fir.char<1,3>>>

    call print_char_array(t_char_array(x=x, c=c1))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_ptr(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>, %[[a:.*]]: !fir.box<!fir.array<?x?xi32>>
  subroutine test_ptr(x, a)
    real :: x
    integer, target :: a(:, :)
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}> {uniq_name = {{.*}}}
    ! CHECK: fir.field_index x, !fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>

    ! CHECK: %[[pfield:.*]] = fir.field_index p, !fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>
    ! CHECK: %[[pcoor:.*]] = fir.coordinate_of %[[tmp]], %[[pfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
    ! CHECK: %[[slice:.*]] = fir.slice %c1{{.*}}, %c4{{.*}}, %c2{{.*}}, %c1{{.*}}, %c3{{.*}}, %c1{{.*}} : (i64, i64, i64, i64, i64, i64) -> !fir.slice<2>
    ! CHECK: %[[rebox:.*]] = fir.rebox %[[a]] [%[[slice]]] : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xi32>>
    ! CHECK: %[[ptr:.*]] = fir.rebox %[[rebox]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
    ! CHECK: fir.store %[[ptr]] to %[[pcoor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
    call print_ptr(t_ptr(x=x, p=a(1:4:2, 1:3:1)))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_nested(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>,
  ! CHECK-SAME: %[[d:.*]]: !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>
  subroutine test_nested(x, d)
    real :: x
    type(t_array) :: d
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}> {uniq_name = {{.*}}}
    ! CHECK: fir.field_index x, !fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>

    ! CHECK: %[[dtfield:.*]] = fir.field_index dt, !fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>
    ! CHECK: %[[dtcoor:.*]] = fir.coordinate_of %[[tmp]], %[[dtfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>>, !fir.field) -> !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>
    ! CHECK: %[[dload:.*]] = fir.load %[[d]] : !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>
    ! CHECK: fir.store %[[dload]] to %[[dtcoor]] : !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>
    call print_nested(t_nested(x=x, dt=d))
  end subroutine

  subroutine print_simple(t)
    type(t_simple) :: t
    print *, t%x
  end subroutine
  subroutine print_char_scalar(t)
    type(t_char_scalar) :: t
    print *, t%x, t%c
  end subroutine
  subroutine print_simple_array(t)
    type(t_array) :: t
    print *, t%x, t%i
  end subroutine
  subroutine print_char_array(t)
    type(t_char_array) :: t
    print *, t%x, t%c
  end subroutine
  subroutine print_ptr(t)
    type(t_ptr) :: t
    print *, t%x, t%p
  end subroutine
  subroutine print_nested(t)
    type(t_nested) :: t
    print *, t%x, t%dt%x, t%dt%i
  end subroutine

end module

  use m_struct_ctor
  integer, target :: i(4,3) = reshape([1,2,3,4,5,6,7,8,9,10,11,12], [4,3])
  call test_simple(42.)
  call test_char_scalar(42.)
  call test_simple_array(42., [1,2,3,4,5])
  call test_char_array(42., ["abc", "def", "geh", "ijk", "lmn"])
  call test_ptr(42., i)
  call test_nested(42., t_array(x=43., i=[5,6,7,8,9]))
end
