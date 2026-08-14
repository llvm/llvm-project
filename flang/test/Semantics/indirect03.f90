! This test checks the OpenMP restriction that the INDIRECT clause on a DECLARE
! TARGET directive is only allowed with DEVICE_TYPE(ANY) (an absent DEVICE_TYPE
! clause also implies ANY). A host- or device-only procedure cannot be the
! target of an indirect device invocation.

! RUN: not %flang -fopenmp -fopenmp-version=52 %s 2>&1 | FileCheck %s

module functions
  implicit none
contains
  !CHECK: Only the DEVICE_TYPE(ANY) clause is allowed with the INDIRECT clause on the DECLARE TARGET directive
  function func_host() result(i)
    !$omp declare target enter(func_host) device_type(host) indirect(.true.)
    character(1) :: i
    i = 'a'
  end function

  !CHECK: Only the DEVICE_TYPE(ANY) clause is allowed with the INDIRECT clause on the DECLARE TARGET directive
  function func_nohost() result(i)
    !$omp declare target enter(func_nohost) device_type(nohost) indirect(.true.)
    character(1) :: i
    i = 'b'
  end function

  ! DEVICE_TYPE(ANY) with INDIRECT is allowed, so no error is expected here.
  function func_any() result(i)
    !$omp declare target enter(func_any) device_type(any) indirect(.true.)
    character(1) :: i
    i = 'c'
  end function

  ! The restriction only applies when INDIRECT evaluates to true, so a
  ! device-only procedure is allowed here.
  function func_false() result(i)
    !$omp declare target enter(func_false) device_type(nohost) indirect(.false.)
    character(1) :: i
    i = 'd'
  end function

  ! The argument may have any logical kind, not just the default one.
  function func_false_kind1() result(i)
    !$omp declare target enter(func_false_kind1) device_type(nohost) indirect(.false._1)
    character(1) :: i
    i = 'e'
  end function

  function func_false_kind8() result(i)
    !$omp declare target enter(func_false_kind8) device_type(nohost) indirect(.false._8)
    character(1) :: i
    i = 'f'
  end function
end module
