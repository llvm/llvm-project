! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

subroutine host_assoc_fail()
  integer :: i
  ! ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp threadprivate(i)
  real :: r
  ! ERROR: A variable that appears in a THREADPRIVATE directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp threadprivate(r)
contains
  subroutine internal()
!$omp parallel
    print *, i, r
!$omp end parallel
  end subroutine internal
end subroutine host_assoc_fail

! This sub-test is not supposed to emit a compiler error.
subroutine host_assoc()
  integer, save :: i
  !$omp threadprivate(i)
  real, save :: r
  !$omp threadprivate(r)
contains
  subroutine internal()
!$omp parallel
    print *, i, r
!$omp end parallel
  end subroutine internal
end subroutine host_assoc
