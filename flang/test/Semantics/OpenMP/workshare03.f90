! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.4 workshare Construct
! All array assignments, scalar assignments, and masked array assignments
! must be intrinsic assignments.

module defined_assign
  interface assignment(=)
    module procedure work_assign
  end interface

  contains
    subroutine work_assign(a,b)
      integer, intent(out) :: a
      logical, intent(in) :: b(:)
    end subroutine work_assign
end module defined_assign

program omp_workshare
  use defined_assign

  integer :: a, aa(10), bb(10)
  logical :: l(10)
  l = .TRUE.

  !$omp workshare
  !ERROR: Defined assignment statement is not allowed in a WORKSHARE construct
  a = l
  aa = bb
  !$omp end workshare

end program omp_workshare
