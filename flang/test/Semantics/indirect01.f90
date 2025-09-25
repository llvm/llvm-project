! This test checks the lowering of OpenMP Indirect Clause when used with the Declare Target directive

! RUN: not flang -fopenmp -fopenmp-version=52 %s 2>&1 | FileCheck %s

module functions
  implicit none

  interface
    function func() result(i)
      character(1) :: i
    end function
  end interface

contains
  function func1() result(i)
    !CHECK: The INDIRECT clause cannot be used without the ENTER clause with the DECLARE TARGET directive.
    !$omp declare target indirect(.true.)
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
