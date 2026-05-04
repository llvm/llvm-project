! RUN: not %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s 2>&1 | FileCheck %s

module m1
  interface operator(.fluffy.)
    procedure my_mul
  end interface
  type t1
    integer :: val = 1
  end type
!$omp declare reduction(.fluffy.:t1:omp_out=omp_out.fluffy.omp_in)
contains
  function my_mul(x, y)
    type (t1), intent (in) :: x, y
    type (t1) :: my_mul
    my_mul%val = x%val * y%val
  end function my_mul

  subroutine subr(a, r)
    implicit none
    integer, intent(in), dimension(10) :: a
    integer, intent(out) :: r
    integer :: i
    !$omp do parallel reduction(.fluffy.:r)
!CHECK: error: The type of 'r' is incompatible with the reduction operator.
    do i=1,10
    end do
  end subroutine subr
end module m1
