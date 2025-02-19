! RUN: bbc -emit-hlfir %s -o - | %python %S/gen_mod_ref_test.py | \
! RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
! RUN:  --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s

! Test fir.call modref with internal procedures

subroutine simple_modref_test(test_var_x)
  implicit none
  real :: test_var_x
  call test_effect_internal()
contains
  subroutine test_effect_internal()
    test_var_x = 0.
  end subroutine
end subroutine
! CHECK-LABEL: Testing : "_QPsimple_modref_test"
! CHECK: test_effect_internal -> test_var_x#0: ModRef

subroutine simple_nomodref_test(test_var_x)
  implicit none
  real :: test_var_x
  call test_effect_internal()
contains
  subroutine test_effect_internal()
    call some_external()
  end subroutine
end subroutine
! CHECK-LABEL: Testing : "_QPsimple_nomodref_test"
! CHECK: test_effect_internal -> test_var_x#0: NoModRef

! Test that effects on captured variable are propagated to associated variables
! in associate construct.

subroutine test_associate()
  implicit none
  real :: test_var_x(10), test_var_a(10)
  associate (test_var_y=>test_var_x)
     test_var_a = test_effect_internal()
  end associate
contains
  function test_effect_internal() result(res)
    real :: res(10)
    res = test_var_x(10:1:-1)
  end function
end subroutine
! CHECK-LABEL: Testing : "_QPtest_associate"
! CHECK: test_effect_internal -> test_var_a#0: NoModRef
! CHECK: test_effect_internal -> test_var_x#0: ModRef
! CHECK: test_effect_internal -> test_var_y#0: ModRef

! Test that captured variables are considered to be affected when calling
! another internal function.
subroutine effect_inside_internal()
  implicit none
  real :: test_var_x(10)
  call internal_sub()
contains
  subroutine internal_sub
    real :: test_var_y(10)
    test_var_y = test_effect_internal_func()
  end subroutine
  function test_effect_internal_func() result(res)
    real :: res(10)
    res = test_var_x(10:1:-1)
  end function
end subroutine
! CHECK-LABEL: Testing : "_QFeffect_inside_internalPinternal_sub"
! CHECK: test_effect_internal_func -> test_var_x#0: ModRef
! CHECK: test_effect_internal_func -> test_var_y#0: NoModRef

! Test that captured variables are considered to be affected when calling
! any procedure
subroutine effect_inside_internal_2()
  implicit none
  real :: test_var_x(10)
  call some_external_that_may_capture_procedure_pointer(capturing_internal_func)
  call internal_sub()
contains
  subroutine internal_sub
    test_var_x(1) = 0
    call test_effect_external_func_may_use_captured_proc_pointer()
  end subroutine
  function capturing_internal_func() result(res)
    real :: res(10)
    res = test_var_x(10:1:-1)
  end function
end subroutine
! CHECK-LABEL: Testing : "_QFeffect_inside_internal_2Pinternal_sub"
! CHECK: test_effect_external_func_may_use_captured_proc_pointer -> test_var_x#0: ModRef

module ifaces
  interface
    subroutine modify_pointer(p)
      real, pointer :: p
    end subroutine
    subroutine modify_allocatable(p)
      real, allocatable :: p
    end subroutine
  end interface
end module

! Test that descriptor address of captured pointer are considered modified
! in internal call.
subroutine test_pointer()
  real, pointer :: test_var_pointer
  call capture_internal(modify_pointer)
  associate (test_var_pointer_target => test_var_pointer)
    ! external may call internal via procedure pointer
    call test_effect_external()
  end associate
contains
  subroutine internal()
    use ifaces, only : modify_pointer
    call modify_pointer(test_var_pointer)
  end subroutine
end subroutine
! CHECK-LABEL: Testing : "_QPtest_pointer"
! CHECK: test_effect_external -> test_var_pointer#0: ModRef
! CHECK: test_effect_external -> test_var_pointer_target#0: ModRef

! Test that descriptor address of captured allocatable are considered modified
! in internal calls.
subroutine test_allocatable()
  real, allocatable :: test_var_allocatable
  call capture_internal(modify_allocatable)
  associate (test_var_allocatable_target => test_var_allocatable)
    ! external may call internal via procedure pointer
    call test_effect_external()
  end associate
contains
  subroutine internal()
    use ifaces, only : modify_allocatable
    call modify_allocatable(test_var_allocatable)
  end subroutine
end subroutine
! CHECK-LABEL: Testing : "_QPtest_allocatable"
! CHECK: test_effect_external -> test_var_allocatable#0: ModRef
! CHECK: test_effect_external -> test_var_allocatable_target#0: ModRef
