! This test checks that the OpenMP INDIRECT clause on the DECLARE TARGET
! directive is rejected before OpenMP 5.1 and accepted from 5.1 onwards.

! RUN: not %flang -fopenmp -fopenmp-version=50 %s 2>&1 | FileCheck %s --check-prefix="CHECK-50"
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix="CHECK-52"

module functions
  implicit none

  interface
    function func() result(i)
      character(1) :: i
    end function
  end interface

contains
  function func1() result(i)
    !CHECK-50: INDIRECT clause is not allowed on DECLARE TARGET directive in OpenMP v5.0, try -fopenmp-version=51
    !CHECK-52: !$OMP DECLARE TARGET ENTER(func1) INDIRECT(.true._4)
    !$omp declare target enter(func1) indirect(.true.)
    character(1) :: i
    i = 'a'
    return
  end function

  ! TO is the pre-5.2 spelling of ENTER, so INDIRECT is accepted with it too.
  function func2() result(i)
    !CHECK-52: !$OMP DECLARE TARGET TO(func2) INDIRECT(.true._4)
    !$omp declare target to(func2) indirect(.true.)
    character(1) :: i
    i = 'b'
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
