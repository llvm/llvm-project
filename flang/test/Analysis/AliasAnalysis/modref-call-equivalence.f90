! RUN: bbc -emit-hlfir %s -o - | %python %S/gen_mod_ref_test.py | \
! RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
! RUN:  --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s

! Test that mod ref effects for variables captured in internal procedures
! propagate to all the variables they are in equivalence with.
subroutine test_captured_equiv()
  implicit none
  real :: test_var_x , test_var_y, test_var_z
  equivalence(test_var_x, test_var_y)
  call test_effect_internal()
contains
subroutine test_effect_internal()
  test_var_y = 0.
end subroutine
end subroutine

! CHECK-LABEL: Testing : "_QPtest_captured_equiv"
! CHECK: test_effect_internal -> test_var_x#0: ModRef
! CHECK: test_effect_internal -> test_var_y#0: ModRef
! CHECK: test_effect_internal -> test_var_z#0: NoModRef

subroutine test_no_capture()
  implicit none
  real :: test_var_x , test_var_y
  equivalence(test_var_x, test_var_y)
  call test_effect_internal()
contains
subroutine test_effect_internal()
end subroutine
end subroutine
! CHECK-LABEL: Testing : "_QPtest_no_capture"
! CHECK: test_effect_internal -> test_var_x#0: NoModRef
! CHECK: test_effect_internal -> test_var_y#0: NoModRef
