! RUN: bbc -emit-hlfir %s -o - | %python %S/gen_mod_ref_test.py | \
! RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
! RUN:  --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s

! Test fir.call modref for global variables (module, saved, common).


module somemod
  implicit none
  real :: test_var_xmod
  interface
    subroutine may_capture(x)
      real, target :: x
    end subroutine
  end interface
end module

subroutine test_module
  use somemod, only : test_var_xmod
  implicit none
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_module"
! CHECK: test_effect_external -> test_var_xmod#0: ModRef

subroutine test_saved_local
  use somemod, only : may_capture
  implicit none
  real, save :: test_var_xsaved
  ! Capture is invalid after the call because test_var_xsaved does not have the
  ! target attribute.
  call may_capture(test_var_xsaved)
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_saved_local"
! CHECK: test_effect_external -> test_var_xsaved#0: NoModRef

subroutine test_saved_target
  use somemod, only : may_capture
  implicit none
  real, save, target :: test_var_target_xsaved
  call may_capture(test_var_target_xsaved)
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_saved_target"
! CHECK: test_effect_external -> test_var_target_xsaved#0: ModRef

subroutine test_saved_target_2
  use somemod, only : may_capture
  implicit none
  real, save, target :: test_var_target_xsaved
  ! Pointer associations made to SAVE variables remain valid after the
  ! procedure exit, so it cannot be ruled out that the variable has been
  ! captured in a previous call to `test_var_target_xsaved` even though the
  ! call to `test_effect_external` appears first here.
  call test_effect_external()
  call may_capture(test_var_target_xsaved)
end subroutine
! CHECK-LABEL: Testing : "_QPtest_saved_target_2"
! CHECK: test_effect_external -> test_var_target_xsaved#0: ModRef

subroutine test_saved_used_in_internal
  implicit none
  real, save :: test_var_saved_captured
  call may_capture_procedure_pointer(internal)
  call test_effect_external()
contains
  subroutine internal
    test_var_saved_captured = 0.
  end subroutine
end subroutine
! CHECK-LABEL: Testing : "_QPtest_saved_used_in_internal"
! CHECK: test_effect_external -> test_var_saved_captured#0: ModRef

subroutine test_common
  implicit none
  real :: test_var_x_common
  common /comm/ test_var_x_common 
  call test_effect_external()
end subroutine
! CHECK-LABEL: Testing : "_QPtest_common"
! CHECK: test_effect_external -> test_var_x_common#0: ModRef
