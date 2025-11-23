! Testing the Semantic failure of forming loop sequences under regular OpenMP directives 

!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine loop_transformation_construct1
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  ! Only 1 do loop is associated with the OMP DO directive so the END DO directive is unmatched
  !$omp do
  do x = 1, i
    v(x) = v(x) * 2
  end do
  do x = 1, i
    v(x) = v(x) * 2
  end do
  !ERROR: Misplaced OpenMP end-directive
  !$omp end do
end subroutine

subroutine loop_transformation_construct2
  implicit none
  integer, parameter :: i = 5
  integer :: x
  integer :: v(i)

  ! Only 1 do loop is associated with the OMP TILE directive so the END TILE directive is unmatched
  !$omp tile sizes(2)
  do x = 1, i
    v(x) = v(x) * 2
  end do
  do x = 1, i
    v(x) = v(x) * 2
  end do
  !ERROR: Misplaced OpenMP end-directive
  !$omp end tile
end subroutine
