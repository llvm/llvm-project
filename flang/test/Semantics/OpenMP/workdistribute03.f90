! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60
! OpenMP Version 6.0
! workdistribute Construct
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

program omp_workdistribute
  use defined_assign

  integer :: a, aa(10), bb(10)
  logical :: l(10)
  l = .TRUE.

  !$omp teams
  !$omp workdistribute
  !ERROR: Defined assignment statement is not allowed in a WORKDISTRIBUTE construct
  a = l
  aa = bb
  !$omp end workdistribute
  !$omp end teams

end program omp_workdistribute
