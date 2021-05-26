! Test lowering of derived type assignments
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Assignment of simple "struct" with trivial intrinsic members.
! CHECK-LABEL: func @_QPtest1
subroutine test1
  type t
     integer a
     integer b
  end type t
  type(t) :: t1, t2
  ! CHECK-DAG: %[[t1:.*]] = fir.alloca !fir.type<_QFtest1Tt{a:i32,b:i32}> {{{.*}}uniq_name = "_QFtest1Et1"}
  ! CHECK-DAG: %[[t2:.*]] = fir.alloca !fir.type<_QFtest1Tt{a:i32,b:i32}> {{{.*}}uniq_name = "_QFtest1Et2"}
  ! CHECK-DAG: %[[a:.*]] = fir.field_index a, !fir.type<_QFtest1Tt{a:i32,b:i32}>
  ! CHECK: %[[ac:.*]] = fir.coordinate_of %[[t2]], %[[a]] : (!fir.ref<!fir.type<_QFtest1Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[ad:.*]] = fir.coordinate_of %[[t1]], %[[a]] : (!fir.ref<!fir.type<_QFtest1Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[ld:.*]] = fir.load %[[ac]] : !fir.ref<i32>
  ! CHECK: fir.store %[[ld]] to %[[ad]] : !fir.ref<i32>
  ! CHECK: %[[b:.*]] = fir.field_index b, !fir.type<_QFtest1Tt{a:i32,b:i32}>
  ! CHECK: %[[bc:.*]] = fir.coordinate_of %[[t2]], %[[b]] : (!fir.ref<!fir.type<_QFtest1Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[bd:.*]] = fir.coordinate_of %[[t1]], %[[b]] : (!fir.ref<!fir.type<_QFtest1Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[ld:.*]] = fir.load %[[bc]] : !fir.ref<i32>
  ! CHECK: fir.store %[[ld]] to %[[bd]] : !fir.ref<i32>
  t1 = t2
end subroutine test1

! Test a defined assignment on a simple struct.
module m2
  type t
     integer a
     integer b
  end type t
  interface assignment (=)
     module procedure t_to_t
  end interface assignment (=)
contains
  ! CHECK-LABEL: func @_QMm2Ptest2
  subroutine test2
    type(t) :: t1, t2
    ! CHECK: fir.call @_QMm2Pt_to_t(%{{.*}}, %{{.*}}) : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>) -> ()
    t1 = t2
    ! CHECK: return
  end subroutine test2

  ! Swap elements on assignment.
  ! CHECK-LABEL: func @_QMm2Pt_to_t(
  ! CHECK-SAME: %[[a1:[^:]*]]: !fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>,
  ! CHECK-SAME: %[[b1:[^:]*]]: !fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>) 
  subroutine t_to_t(a1,b1)
    type(t), intent(out) :: a1
    type(t), intent(in) :: b1
    ! CHECK: %[[a:.*]] = fir.field_index a, !fir.type<_QMm2Tt{a:i32,b:i32}>
    ! CHECK: %[[a1a:.*]] = fir.coordinate_of %[[a1]], %[[a]] : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK: %[[b:.*]] = fir.field_index b, !fir.type<_QMm2Tt{a:i32,b:i32}>
    ! CHECK: %[[b1b:.*]] = fir.coordinate_of %[[b1]], %[[b]] : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK: %[[v:.*]] = fir.load %[[b1b]] : !fir.ref<i32>
    ! CHECK: fir.store %[[v]] to %[[a1a]] : !fir.ref<i32>
    ! CHECK: %[[b:.*]] = fir.field_index b, !fir.type<_QMm2Tt{a:i32,b:i32}>
    ! CHECK: %[[a1b:.*]] = fir.coordinate_of %[[a1]], %[[b]] : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK: %[[a:.*]] = fir.field_index a, !fir.type<_QMm2Tt{a:i32,b:i32}>
    ! CHECK: %[[b1a:.*]] = fir.coordinate_of %[[b1]], %[[a]] : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK: %[[v:.*]] = fir.load %[[b1a]] : !fir.ref<i32>
    ! CHECK: fir.store %[[v]] to %[[a1b]] : !fir.ref<i32>
    a1%a = b1%b
    a1%b = b1%a
    ! CHECK: return
  end subroutine t_to_t
end module m2

! CHECK-LABEL: func @_QPtest3
subroutine test3
  type t
     character(LEN=20) :: m_c
     integer :: m_i
  end type t
  type(t) :: t1, t2
  ! CHECK-DAG: %[[t1:.*]] = fir.alloca !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}> {{{.*}}uniq_name = "_QFtest3Et1"}
  ! CHECK-DAG: %[[t2:.*]] = fir.alloca !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}> {{{.*}}uniq_name = "_QFtest3Et2"}

  ! CHECK: %[[mc:.*]] = fir.field_index m_c, !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>
  ! CHECK: %[[t2x:.*]] = fir.coordinate_of %[[t2]], %[[mc]] : (!fir.ref<!fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.char<1,20>>
  ! CHECK: %[[t1x:.*]] = fir.coordinate_of %[[t1]], %[[mc]] : (!fir.ref<!fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.char<1,20>>
  ! CHECK-DAG: %[[one:.*]] = constant 1
  ! CHECK: %[[count:.*]] = muli %[[one]], %
  ! CHECK: constant false
  ! CHECK: %[[dst:.*]] = fir.convert %[[t1x]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
  ! CHECK: %[[src:.*]] = fir.convert %[[t2x]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0i8.p0i8.i64(%[[dst]], %[[src]], %[[count]], %false) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()


  ! CHECK: %[[mi:.*]] = fir.field_index m_i, !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>
  ! CHECK: %[[mip:.*]] = fir.coordinate_of %[[t1]], %[[mi]] : (!fir.ref<!fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[ii:.*]] = fir.load
  ! CHECK: fir.store %[[ii]] to %[[mip]] : !fir.ref<i32>
  t1 = t2
  ! CHECK: return
end subroutine test3

! CHECK-LABEL: func @_QPtest_array_comp(
! CHECK-SAME: %[[t1:.*]]: !fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>,
! CHECK-SAME: %[[t2:.*]]: !fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>)
subroutine test_array_comp(t1, t2)
  type t
     real :: m_x(10)
     integer :: m_i
  end type t
  type(t) :: t1, t2
  ! CHECK: %[[xfield:.*]] = fir.field_index m_x, !fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>
  ! CHECK-DAG: %[[x2coor:.*]] = fir.coordinate_of %[[t2]], %[[xfield]] : (!fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
  ! CHECK-DAG: %[[x1coor:.*]] = fir.coordinate_of %[[t1]], %[[xfield]] : (!fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
  ! CHECK-DAG: %[[x1load:.*]] = fir.array_load %[[x1coor]](%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK-DAG: %[[x2load:.*]] = fir.array_load %[[x2coor]](%{{.*}}) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.array<10xf32>
  ! CHECK: %[[loop:.*]] = fir.do_loop %[[idx:.*]] = %c0{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%[[res:.*]] = %[[x1load]]) -> (!fir.array<10xf32>) {
  ! CHECK:   %[[fetch:.*]] = fir.array_fetch %[[x2load]], %[[idx]] : (!fir.array<10xf32>, index) -> f32
  ! CHECK:   %[[update:.*]] = fir.array_update %[[res]], %[[fetch]], %[[idx]] : (!fir.array<10xf32>, f32, index) -> !fir.array<10xf32>
  ! CHECK:   fir.result %[[update]] : !fir.array<10xf32>
  ! CHECK: fir.array_merge_store %[[x1load]], %[[loop]] to %[[x1coor]] : !fir.ref<!fir.array<10xf32>>

  ! CHECK: fir.field_index m_i, !fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>
  t1 = t2
end subroutine

! CHECK-LABEL: func @_QPtest_ptr_comp(
! CHECK-SAME: %[[t1:.*]]: !fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>,
! CHECK-SAME: %[[t2]]: !fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>)
subroutine test_ptr_comp(t1, t2)
  type t
     complex, pointer :: ptr(:)
     integer :: m_i
  end type t
  type(t) :: t1, t2
  ! CHECK: %[[ptrfield:.*]] = fir.field_index ptr, !fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>
  ! CHECK-DAG: %[[ptr2coor:.*]] = fir.coordinate_of %[[t2]], %[[ptrfield]] : (!fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
  ! CHECK: %[[ptr1coor:.*]] = fir.coordinate_of %[[t1]], %[[ptrfield]] : (!fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
  ! CHECK-DAG: %[[ptr:.*]] = fir.load %[[ptr2coor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
  ! CHECK: fir.store %[[ptr]] to %[[ptr1coor]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>

  ! CHECK: fir.field_index m_i, !fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>
  t1 = t2
end subroutine
