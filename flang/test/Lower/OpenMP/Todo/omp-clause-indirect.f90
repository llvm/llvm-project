! This test checks the lowering of OpenMP Indirect Clause when used with the Declare Target directive

! RUN: not %flang_fc1 -emit-fir -fopenmp -fopenmp-version=52 %s 2>&1 | FileCheck %s

module functions
  implicit none

  interface
    function func() result(i)
      character(1) :: i
    end function
  end interface

contains
  function func1() result(i)
    !CHECK: not yet implemented: Unhandled clause INDIRECT in DECLARE TARGET construct
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

  !$omp target map(from: val1)
  val1 = ptr1()
  !$omp end target

end program
