! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test that TYPEOF and CLASSOF type specifiers from F2023 lower correctly.
! Semantics resolves TYPEOF/CLASSOF to the concrete type of the referenced
! object, so lowering should produce the same FIR types as explicit type
! declarations.

module typeof_classof_types
  type :: base_t
    integer :: x
  end type
  type, extends(base_t) :: child_t
    integer :: y
  end type
contains

! Test TYPEOF with intrinsic types
  subroutine test_typeof_integer(a)
    integer :: a
    typeof(a) :: b
    b = a
  end subroutine
! CHECK-LABEL: func.func @_QMtypeof_classof_typesPtest_typeof_integer(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32>
! CHECK: %[[B:.*]] = fir.alloca i32 {bindc_name = "b"
! CHECK: hlfir.declare %[[B]]

  subroutine test_typeof_real8(a)
    real(8) :: a
    typeof(a) :: b
    b = a
  end subroutine
! CHECK-LABEL: func.func @_QMtypeof_classof_typesPtest_typeof_real8(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<f64>
! CHECK: %[[B:.*]] = fir.alloca f64 {bindc_name = "b"

  subroutine test_typeof_logical(a)
    logical :: a
    typeof(a) :: b
    b = a
  end subroutine
! CHECK-LABEL: func.func @_QMtypeof_classof_typesPtest_typeof_logical(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.logical<4>>
! CHECK: %[[B:.*]] = fir.alloca !fir.logical<4> {bindc_name = "b"

! Test TYPEOF with derived types
  subroutine test_typeof_derived(a)
    type(base_t) :: a
    typeof(a) :: b
    b = a
  end subroutine
! CHECK-LABEL: func.func @_QMtypeof_classof_typesPtest_typeof_derived(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QMtypeof_classof_typesTbase_t{x:i32}>>
! CHECK: %[[B:.*]] = fir.alloca !fir.type<_QMtypeof_classof_typesTbase_t{x:i32}> {bindc_name = "b"

  subroutine test_typeof_child(a)
    type(child_t) :: a
    typeof(a) :: b
    b = a
  end subroutine
! CHECK-LABEL: func.func @_QMtypeof_classof_typesPtest_typeof_child(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.type<_QMtypeof_classof_typesTchild_t{base_t:!fir.type<_QMtypeof_classof_typesTbase_t{x:i32}>,y:i32}>>
! CHECK: %[[B:.*]] = fir.alloca !fir.type<_QMtypeof_classof_typesTchild_t{base_t:!fir.type<_QMtypeof_classof_typesTbase_t{x:i32}>,y:i32}> {bindc_name = "b"

! Test CLASSOF with allocatable (polymorphic)
  subroutine test_classof_allocatable(a)
    class(base_t), intent(in) :: a
    classof(a), allocatable :: b
    allocate(b, source=a)
  end subroutine
! CHECK-LABEL: func.func @_QMtypeof_classof_typesPtest_classof_allocatable(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMtypeof_classof_typesTbase_t{x:i32}>>
! CHECK: %[[B:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMtypeof_classof_typesTbase_t{x:i32}>>> {bindc_name = "b"

! Test CLASSOF with pointer (polymorphic)
  subroutine test_classof_pointer(a)
    class(base_t), target, intent(in) :: a
    classof(a), pointer :: b
    b => a
  end subroutine
! CHECK-LABEL: func.func @_QMtypeof_classof_typesPtest_classof_pointer(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMtypeof_classof_typesTbase_t{x:i32}>>
! CHECK: %[[B:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.type<_QMtypeof_classof_typesTbase_t{x:i32}>>> {bindc_name = "b"

end module
