! RUN: bbc -emit-hlfir %s -o - | %python %S/gen_mod_ref_test.py | \
! RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
! RUN:  --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s

! Test fir.call modref when arguments are passed to the call. This focus
! on the possibility of "direct" effects (taken via the arguments, and not
! via some indirect access via global states).

subroutine test_simple()
  implicit none
  real :: test_var_x, test_var_y
  call test_effect_external(test_var_x)
end subroutine
! CHECK-LABEL: Testing : "_QPtest_simple"
! CHECK: test_effect_external -> test_var_x#0: ModRef
! CHECK: test_effect_external -> test_var_y#0: NoModRef

subroutine test_equivalence()
  implicit none
  real :: test_var_x, test_var_y
  equivalence(test_var_x, test_var_y)
  call test_effect_external(test_var_x)
end subroutine
! CHECK-LABEL: Testing : "_QPtest_equivalence"
! CHECK: test_effect_external -> test_var_x#0: ModRef
! CHECK: test_effect_external -> test_var_y#0: ModRef

subroutine test_pointer()
  implicit none
  real, target :: test_var_x, test_var_y
  real, pointer :: p
  p => test_var_x
  call test_effect_external(p)
end subroutine
! CHECK-LABEL: Testing : "_QPtest_pointer"
! CHECK: test_effect_external -> test_var_x#0: ModRef
! TODO: test_var_y should be NoModRef, the alias analysis is currently very
! conservative whenever pointer/allocatable descriptors are involved (mostly
! because it needs to make sure it is dealing descriptors for POINTER/ALLOCATABLE
! from the Fortran source and that it can apply language rules).
! CHECK: test_effect_external -> test_var_y#0: ModRef

subroutine test_array_1(test_var_x)
  implicit none
  real :: test_var_x(:), test_var_y
  call test_effect_external(test_var_x(10))
end subroutine
! CHECK-LABEL: Testing : "_QPtest_array_1"
! CHECK: test_effect_external -> test_var_x#0: ModRef
! CHECK: test_effect_external -> test_var_y#0: NoModRef

subroutine test_array_copy_in(test_var_x)
  implicit none
  real :: test_var_x(:), test_var_y
  call test_effect_external_2(test_var_x)
end subroutine
! CHECK-LABEL: Testing : "_QPtest_array_copy_in"
! CHECK: test_effect_external_2 -> test_var_x#0: ModRef
! TODO: copy-in/out is currently badly understood by alias analysis, this
! causes the modref analysis to think the argument may alias with anyting.
! test_var_y should obviously be considered NoMoRef in the call.
! CHECK: test_effect_external_2 -> test_var_y#0: ModRef
