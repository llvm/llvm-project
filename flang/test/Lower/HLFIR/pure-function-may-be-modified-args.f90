! Test that the arguments of PURE functions that may be modified (INTENT(OUT)
! and INTENT(INOUT) dummy arguments, an extension) are marked with
! fir.may_be_modified on the call.
! RUN: %flang_fc1 -emit-hlfir -o - %s 2>&1 | FileCheck %s

module may_be_modified_m
  type t
  contains
    procedure, nopass :: tbp => setstat
    procedure :: tbp_pass
  end type
contains
  pure integer function setstat(x, s)
    integer, intent(in) :: x
    integer, intent(inout) :: s
    s = 0
    setstat = x
  end function
  pure integer function tbp_pass(this, s, v)
    class(t), intent(in) :: this
    integer, intent(out) :: s
    integer, value :: v
    s = v
    tbp_pass = v
  end function
  pure integer function readonly(x, s)
    integer, intent(in) :: x, s
    readonly = x + s
  end function
  integer function impure_inout(x, s)
    integer, intent(in) :: x
    integer, intent(inout) :: s
    s = 0
    impure_inout = x
  end function
  pure subroutine pure_sub(s)
    integer, intent(inout) :: s
    s = 0
  end subroutine
  pure integer function pointer_dummy(p)
    integer, pointer :: p
    pointer_dummy = p
  end function
end module

! External procedures defined in the same file and called with implicit
! interfaces.
pure integer function ext_readonly(x, s)
  integer, intent(in) :: x, s
  ext_readonly = x + s
end function
pure integer function ext_setstat(x, s)
  integer, intent(in) :: x
  integer, intent(inout) :: s
  s = 0
  ext_setstat = x
end function

subroutine test_direct(x, s, r)
  use may_be_modified_m
  integer :: x, s, r
  r = setstat(x, s)
end subroutine
! CHECK-LABEL: func.func @_QPtest_direct(
! CHECK: fir.call @_QMmay_be_modified_mPsetstat(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32> {fir.may_be_modified}) -> i32

subroutine test_pointer(x, s, r)
  use may_be_modified_m
  procedure(setstat), pointer :: p
  integer :: x, s, r
  p => setstat
  r = p(x, s)
end subroutine
! CHECK-LABEL: func.func @_QPtest_pointer(
! CHECK: fir.call %{{.*}}(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32> {fir.may_be_modified}) -> i32

subroutine test_dispatch(o, x, s, r)
  use may_be_modified_m
  class(t) :: o
  integer :: x, s, r
  r = o%tbp(x, s)
  r = o%tbp_pass(s, x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_dispatch(
! CHECK: fir.dispatch "tbp"(%{{.*}} : !fir.class<!fir.type<_QMmay_be_modified_mTt>>) (%{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<i32>) -> i32 proc_attrs <pure> arg_attrs([{}, {fir.may_be_modified}])
! CHECK: fir.dispatch "tbp_pass"(%{{.*}} : !fir.class<!fir.type<_QMmay_be_modified_mTt>>) (%{{.*}}, %{{.*}}, %{{.*}} : !fir.class<!fir.type<_QMmay_be_modified_mTt>>, !fir.ref<i32>, i32) -> i32 proc_attrs <pure> pass_arg_pos(0) arg_attrs([{}, {fir.may_be_modified}, {}])

subroutine test_implicit_interface(x, s, r)
  implicit none
  integer :: x, s, r
  r = ext_setstat(x, s)
end subroutine
! CHECK-LABEL: func.func @_QPtest_implicit_interface(
! CHECK: fir.call @_QPext_setstat(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32> {fir.may_be_modified}) -> i32

subroutine test_not_marked(x, s, r, p)
  use may_be_modified_m
  implicit none
  integer :: x, s, r
  integer, pointer :: p
  r = readonly(x, s)
  r = impure_inout(x, s)
  call pure_sub(s)
  r = ext_readonly(x, s)
  r = pointer_dummy(p)
end subroutine
! CHECK-LABEL: func.func @_QPtest_not_marked(
! CHECK: fir.call @_QMmay_be_modified_mPreadonly(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> i32
! CHECK: fir.call @_QMmay_be_modified_mPimpure_inout(%{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> i32
! CHECK: fir.call @_QMmay_be_modified_mPpure_sub(%{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>) -> ()
! CHECK: fir.call @_QPext_readonly(%{{.*}}, %{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>, !fir.ref<i32>) -> i32
! CHECK: fir.call @_QMmay_be_modified_mPpointer_dummy(%{{.*}}) proc_attrs<pure> fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> i32
