! This test checks the lowering of the OpenMP INDIRECT clause when used with the
! DECLARE TARGET directive, together with an indirect call (through a procedure
! pointer) from within a target region.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-target-device %s -o - | FileCheck %s

module functions
  implicit none

  interface
    function func() result(i)
      character(1) :: i
    end function
  end interface

contains
  ! CHECK: func.func @_QMfunctionsPfunc1({{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), automap = false, indirect = true>}
  function func1() result(i)
    !$omp declare target enter(func1) indirect(.true.)
    character(1) :: i
    i = 'a'
    return
  end function
end module

program main
  use functions
  implicit none
  procedure (func), pointer :: ptr1=>func1
  character(1) :: val1

  ! CHECK-LABEL: func.func @_QQmain()
  ! CHECK: omp.target
  !$omp target map(from: val1)
  val1 = ptr1()
  !$omp end target

end program
