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
  ! CHECK: func.func @_QMfunctionsPfunc1({{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter), indirect = true>}
  function func1() result(i)
    !$omp declare target enter(func1) indirect(.true.)
    character(1) :: i
    i = 'a'
    return
  end function

  ! The argument may have any logical kind, not just the default one. A false
  ! value matches the attribute default, so no `indirect` field is printed.
  ! CHECK: func.func @_QMfunctionsPfunc2({{.*}}capture_clause = (enter)>}
  function func2() result(i)
    !$omp declare target enter(func2) indirect(.false._1)
    character(1) :: i
    i = 'b'
  end function

  ! CHECK: func.func @_QMfunctionsPfunc3({{.*}}capture_clause = (enter)>}
  function func3() result(i)
    !$omp declare target enter(func3) indirect(.false._8)
    character(1) :: i
    i = 'c'
  end function
end module

program main
  use functions
  implicit none
  procedure (func), pointer :: ptr1=>func1
  character(1) :: val1

  ! CHECK-LABEL: func.func @_QQmain()
  ! CHECK: omp.target
  ! The procedure pointer is resolved to a callable address that is then used
  ! as the callee of an indirect fir.call.
  ! CHECK: %[[PROC:.*]] = fir.load %{{.*}} : !fir.ref<!fir.boxproc<{{.*}}>>
  ! CHECK: %[[CALLEE:.*]] = fir.box_addr %[[PROC]] : (!fir.boxproc<{{.*}}>) -> {{.*}}
  ! CHECK: fir.call %[[CALLEE]](
  !$omp target map(from: val1)
  val1 = ptr1()
  !$omp end target

end program
