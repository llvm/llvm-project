! RUN: bbc -emit-hlfir %s -o - | %python %S/gen_mod_ref_test.py | \
! RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
! RUN:  --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s

! Test fir.call modref for dummy argument variables. This focus on
! the possibility of indirect effects inside the call.

module somemod
  interface
    subroutine may_capture(x)
      real, target :: x
    end subroutine
    subroutine set_pointer(x)
      real, pointer :: x
    end subroutine
  end interface
end module

subroutine test_dummy(test_var_x)
  use somemod, only : may_capture
  implicit none
  real :: test_var_x
  ! Capture is invalid after the call because test_var_x does not have the
  ! target attribute.
  call may_capture(test_var_x)
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_dummy"
! CHECK: test_effect_external -> test_var_x#0: NoModRef

subroutine test_dummy_target(test_var_x_target)
  use somemod, only : may_capture
  implicit none
  real, target :: test_var_x_target
  call may_capture(test_var_x_target)
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_dummy_target"
! CHECK: test_effect_external -> test_var_x_target#0: ModRef

subroutine test_dummy_pointer(p)
  use somemod, only : set_pointer
  implicit none
  real, pointer :: p
  call set_pointer(p)
  ! Use associated to test the pointer target address, no the
  ! address of the pointer descriptor.
  associate(test_var_p_target  => p)
    call test_effect_external()
  end associate
end subroutine
! CHECK-LABEL: Testing : "_QPtest_dummy_pointer"
! CHECK-DAG: test_effect_external -> test_var_p_target#0: ModRef
! CHECK-DAG: test_effect_external -> box_addr_0#0: ModRef

subroutine test_dummy_allocatable(test_var_x)
  use somemod, only : may_capture
  implicit none
  real, allocatable :: test_var_x
  ! Capture is invalid after the call because test_var_x does not have the
  ! target attribute.
  call may_capture(test_var_x)
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_dummy_allocatable"
! CHECK-DAG: test_effect_external -> test_var_x#0: NoModRef
! We used to report the next one as ModRef:
! CHECK-DAG: test_effect_external -> box_addr_1#0: NoModRef

subroutine test_target_dummy_allocatable(test_var_x)
  use somemod, only : may_capture
  implicit none
  real, allocatable, target :: test_var_x
  call may_capture(test_var_x)
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_target_dummy_allocatable"
! CHECK-DAG: test_effect_external -> test_var_x#0: ModRef
! We used to report the next one as ModRef:
! CHECK-DAG: test_effect_external -> box_addr_2#0: ModRef

subroutine test_dummy_derived_with_allocatable(test_var_x)
  use somemod, only : may_capture
  implicit none
  type t
     real, allocatable :: member
  end type t
  type(t) test_var_x
  call may_capture(test_var_x%member)
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_dummy_derived_with_allocatable"
! CHECK-DAG: test_effect_external -> test_var_x#0: NoModRef
! We used to report the next one as ModRef:
! CHECK-DAG: test_effect_external -> box_addr_3#0: NoModRef

subroutine test_target_dummy_derived_with_allocatable(test_var_x)
  use somemod, only : may_capture
  implicit none
  type t
     real, allocatable :: member
  end type t
  type(t), target :: test_var_x
  call may_capture(test_var_x%member)
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_target_dummy_derived_with_allocatable"
! CHECK-DAG: test_effect_external -> test_var_x#0: ModRef
! We used to report the next one as ModRef:
! CHECK-DAG: test_effect_external -> box_addr_4#0: ModRef
