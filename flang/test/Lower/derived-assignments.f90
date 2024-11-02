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
  ! CHECK-DAG:  %[[VAL_0:.*]] = fir.alloca !fir.type<_QFtest1Tt{a:i32,b:i32}> {{{.*}}uniq_name = "_QFtest1Et1"}
  ! CHECK-DAG:  %[[VAL_1:.*]] = fir.alloca !fir.type<_QFtest1Tt{a:i32,b:i32}> {{{.*}}uniq_name = "_QFtest1Et2"}
  ! CHECK:  %[[VAL_2:.*]] = fir.field_index a, !fir.type<_QFtest1Tt{a:i32,b:i32}>
  ! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QFtest1Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_2b:.*]] = fir.field_index a, !fir.type<_QFtest1Tt{a:i32,b:i32}>
  ! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2b]] : (!fir.ref<!fir.type<_QFtest1Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
  ! CHECK:  fir.store %[[VAL_5]] to %[[VAL_4]] : !fir.ref<i32>
  ! CHECK:  %[[VAL_6:.*]] = fir.field_index b, !fir.type<_QFtest1Tt{a:i32,b:i32}>
  ! CHECK:  %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.type<_QFtest1Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_6b:.*]] = fir.field_index b, !fir.type<_QFtest1Tt{a:i32,b:i32}>
  ! CHECK:  %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_6b]] : (!fir.ref<!fir.type<_QFtest1Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
  ! CHECK:  fir.store %[[VAL_9]] to %[[VAL_8]] : !fir.ref<i32>
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
  ! CHECK-SAME: %[[a1:[^:]*]]: !fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>{{.*}}, %[[b1:[^:]*]]: !fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>{{.*}}) {
  subroutine t_to_t(a1,b1)
    type(t), intent(out) :: a1
    type(t), intent(in) :: b1
    ! CHECK: %[[b:.*]] = fir.field_index b, !fir.type<_QMm2Tt{a:i32,b:i32}>
    ! CHECK: %[[b1b:.*]] = fir.coordinate_of %[[b1]], %[[b]] : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK: %[[v:.*]] = fir.load %[[b1b]] : !fir.ref<i32>
    ! CHECK: %[[a:.*]] = fir.field_index a, !fir.type<_QMm2Tt{a:i32,b:i32}>
    ! CHECK: %[[a1a:.*]] = fir.coordinate_of %[[a1]], %[[a]] : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK: fir.store %[[v]] to %[[a1a]] : !fir.ref<i32>
    ! CHECK: %[[a:.*]] = fir.field_index a, !fir.type<_QMm2Tt{a:i32,b:i32}>
    ! CHECK: %[[b1a:.*]] = fir.coordinate_of %[[b1]], %[[a]] : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
    ! CHECK: %[[v:.*]] = fir.load %[[b1a]] : !fir.ref<i32>
    ! CHECK: %[[b:.*]] = fir.field_index b, !fir.type<_QMm2Tt{a:i32,b:i32}>
    ! CHECK: %[[a1b:.*]] = fir.coordinate_of %[[a1]], %[[b]] : (!fir.ref<!fir.type<_QMm2Tt{a:i32,b:i32}>>, !fir.field) -> !fir.ref<i32>
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
  ! CHECK-DAG:  %[[VAL_0:.*]] = fir.alloca !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}> {{{.*}}uniq_name = "_QFtest3Et1"}
  ! CHECK-DAG:  %[[VAL_1:.*]] = fir.alloca !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}> {{{.*}}uniq_name = "_QFtest3Et2"}
  ! CHECK:  %[[VAL_2:.*]] = fir.field_index m_c, !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>
  ! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.char<1,20>>
  ! CHECK:  %[[VAL_2b:.*]] = fir.field_index m_c, !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>
  ! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2b]] : (!fir.ref<!fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.char<1,20>>
  ! CHECK:  %[[VAL_5:.*]] = arith.constant 20 : index
  ! CHECK:  %[[VAL_6:.*]] = arith.constant 1 : i64
  ! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (index) -> i64
  ! CHECK:  %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_7]] : i64
  ! CHECK:  %[[VAL_9:.*]] = arith.constant false
  ! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
  ! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<i8>
  ! CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_10]], %[[VAL_11]], %[[VAL_8]], %[[VAL_9]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:  %[[VAL_12:.*]] = fir.field_index m_i, !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>
  ! CHECK:  %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_12]] : (!fir.ref<!fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_12b:.*]] = fir.field_index m_i, !fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>
  ! CHECK:  %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_12b]] : (!fir.ref<!fir.type<_QFtest3Tt{m_c:!fir.char<1,20>,m_i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_15:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
  ! CHECK:  fir.store %[[VAL_15]] to %[[VAL_14]] : !fir.ref<i32>
  t1 = t2
  ! CHECK: return
end subroutine test3

! CHECK-LABEL: func @_QPtest_array_comp(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>{{.*}}) {
subroutine test_array_comp(t1, t2)
  type t
     real :: m_x(10)
     integer :: m_i
  end type t
  type(t) :: t1, t2

  ! CHECK:  %[[VAL_2:.*]] = fir.field_index m_x, !fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>
  ! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
  ! CHECK:  %[[VAL_2b:.*]] = fir.field_index m_x, !fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>
  ! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2b]] : (!fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
  ! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : index
  ! CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
  ! CHECK:  %[[VAL_7:.*]] = arith.constant 9 : index
  ! CHECK:  fir.do_loop %[[VAL_8:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_6]] {
  ! CHECK:    %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_4]], %[[VAL_8]] : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
  ! CHECK:    %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_8]] : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
  ! CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<f32>
  ! CHECK:    fir.store %[[VAL_11]] to %[[VAL_9]] : !fir.ref<f32>
  ! CHECK:  }
  ! CHECK:  %[[VAL_12:.*]] = fir.field_index m_i, !fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>
  ! CHECK:  %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_12]] : (!fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_12b:.*]] = fir.field_index m_i, !fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>
  ! CHECK:  %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_12b]] : (!fir.ref<!fir.type<_QFtest_array_compTt{m_x:!fir.array<10xf32>,m_i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_15:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
  ! CHECK:  fir.store %[[VAL_15]] to %[[VAL_14]] : !fir.ref<i32>
  t1 = t2
end subroutine

! CHECK-LABEL: func @_QPtest_ptr_comp(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>{{.*}}, %[[VAL_1]]: !fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>{{.*}}) {
subroutine test_ptr_comp(t1, t2)
  type t
     complex, pointer :: ptr(:)
     integer :: m_i
  end type t
  type(t) :: t1, t2

  ! CHECK:  %[[VAL_2:.*]] = fir.field_index ptr, !fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>
  ! CHECK:  %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
  ! CHECK:  %[[VAL_2b:.*]] = fir.field_index ptr, !fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>
  ! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2b]] : (!fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
  ! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
  ! CHECK:  fir.store %[[VAL_5]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>>
  ! CHECK:  %[[VAL_6:.*]] = fir.field_index m_i, !fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>
  ! CHECK:  %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_6b:.*]] = fir.field_index m_i, !fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>
  ! CHECK:  %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_6b]] : (!fir.ref<!fir.type<_QFtest_ptr_compTt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.complex<4>>>>,m_i:i32}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
  ! CHECK:  fir.store %[[VAL_9]] to %[[VAL_8]] : !fir.ref<i32>
  t1 = t2
end subroutine

! CHECK-LABEL: func @_QPtest_box_assign(
! CHECK-SAME: %[[t1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_box_assignTt{i:i32}>>>>{{.*}}, %[[t2:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_box_assignTt{i:i32}>>>>{{.*}}) {
subroutine test_box_assign(t1, t2)
  type t
     integer :: i
  end type t
  ! Note: the implementation of this case is not optimal, the runtime call is overkill, but right now
  ! lowering is conservative with derived type pointers because it does not make a difference between the
  ! polymorphic and non polymorphic ones at the FIR level.
  type(t), pointer :: t1, t2
  ! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QFtest_box_assignTt{i:i32}>>>
  ! CHECK: %[[t2Load:.*]] = fir.load %[[t2]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_box_assignTt{i:i32}>>>>
  ! CHECK: %[[t1Load:.*]] = fir.load %[[t1]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_box_assignTt{i:i32}>>>>
  ! CHECK: fir.store %[[t1Load]] to %[[tmpBox]] : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_box_assignTt{i:i32}>>>>
  ! CHECK: %[[lhs:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QFtest_box_assignTt{i:i32}>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[rhs:.*]] = fir.convert %[[t2Load]] : (!fir.box<!fir.ptr<!fir.type<_QFtest_box_assignTt{i:i32}>>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAAssign(%[[lhs]], %[[rhs]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  t1 = t2
end subroutine

! CHECK-LABEL: func @_QPtest_alloc_comp(
! CHECK-SAME: %[[t1:.*]]: !fir.ref<!fir.type<_QFtest_alloc_compTt{x:!fir.box<!fir.heap<!fir.array<?x?xf32>>>,i:i32}>>{{.*}}, %[[t2:.*]]: !fir.ref<!fir.type<_QFtest_alloc_compTt{x:!fir.box<!fir.heap<!fir.array<?x?xf32>>>,i:i32}>>{{.*}}) {
subroutine test_alloc_comp(t1, t2)
! Test that derived type assignment with allocatable components are using the
! runtime to handle the deep copy.
  type t
    real, allocatable :: x(:, :)
    integer :: i
  end type
  type(t) :: t1, t2
  ! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.type<_QFtest_alloc_compTt{{.*}}>>
  ! CHECK: %[[t1Box:.*]] = fir.embox %[[t1]] : (!fir.ref<!fir.type<_QFtest_alloc_compTt{{.*}}>>) -> !fir.box<!fir.type<_QFtest_alloc_compTt{{.*}}>>
  ! CHECK: %[[t2Box:.*]] = fir.embox %[[t2]] : (!fir.ref<!fir.type<_QFtest_alloc_compTt{{.*}}>>) -> !fir.box<!fir.type<_QFtest_alloc_compTt{{.*}}>>
  ! CHECK: fir.store %[[t1Box]] to %[[tmpBox]] : !fir.ref<!fir.box<!fir.type<_QFtest_alloc_compTt{{.*}}>>>
  ! CHECK: %[[lhs:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.type<_QFtest_alloc_compTt{{.*}}>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[rhs:.*]] = fir.convert %[[t2Box]] : (!fir.box<!fir.type<_QFtest_alloc_compTt{{.*}}>>) -> !fir.box<none>
  ! CHECK: fir.call @_FortranAAssign(%[[lhs]], %[[rhs]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  t1 = t2
end subroutine

! Reinstate this test when polymorphic types are more fully supported.
!
!module component_with_user_def_assign
!  type t0
!    integer :: i
!    integer :: j
!  contains
!    procedure :: user_def
!    generic :: assignment(=) => user_def
!  end type
!  interface
!  subroutine user_def(other, self)
!    import t0
!    class(t0), intent(out) :: other
!    class(t0), intent(in) :: self
!  end subroutine
!  end interface

!  ! Assignments of type(t) must call the user defined assignment for component a.
!  ! Currently this is delegated to the runtime.
!  type t
!    type(t0) :: a
!    integer :: i
!  end type

!contains
!  ! cHECK-LABEL: func @_QMcomponent_with_user_def_assignPtest(
!  ! cHECK-SAME: %[[t1:.*]]: !fir.ref<!fir.type<_QMcomponent_with_user_def_assignTt{a:!fir.type<_QMcomponent_with_user_def_assignTt0{i:i32,j:i32}>,i:i32}>>{{.*}}, %[[t2:.*]]: !fir.ref<!fir.type<_QMcomponent_with_user_def_assignTt{a:!fir.type<_QMcomponent_with_user_def_assignTt0{i:i32,j:i32}>,i:i32}>>{{.*}}) {
!  subroutine test(t1, t2)
!    type(t) :: t1, t2
!    ! cHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.type<_QMcomponent_with_user_def_assignTt{{.*}}>>
!    ! cHECK: %[[t1Box:.*]] = fir.embox %[[t1]] : (!fir.ref<!fir.type<_QMcomponent_with_user_def_assignTt{{.*}}>>) -> !fir.box<!fir.type<_QMcomponent_with_user_def_assignTt{{.*}}>>
!    ! cHECK: %[[t2Box:.*]] = fir.embox %[[t2]] : (!fir.ref<!fir.type<_QMcomponent_with_user_def_assignTt{{.*}}>>) -> !fir.box<!fir.type<_QMcomponent_with_user_def_assignTt{{.*}}>>
!    ! cHECK: fir.store %[[t1Box]] to %[[tmpBox]] : !fir.ref<!fir.box<!fir.type<_QMcomponent_with_user_def_assignTt{{.*}}>>>
!    ! cHECK: %[[lhs:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.type<_QMcomponent_with_user_def_assignTt{{.*}}>>>) -> !fir.ref<!fir.box<none>>
!    ! cHECK: %[[rhs:.*]] = fir.convert %[[t2Box]] : (!fir.box<!fir.type<_QMcomponent_with_user_def_assignTt{{.*}}>>) -> !fir.box<none>
!    ! cHECK: fir.call @_FortranAAssign(%[[lhs]], %[[rhs]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
!    t1 = t2
!  end subroutine
!end module
