!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50

subroutine f00
  real :: x
!ERROR: The type of 'x' is incompatible with the reduction operator.
!$omp taskgroup task_reduction(.or.: x)
!$omp end taskgroup
end

subroutine f01
  real :: x
!ERROR: Invalid reduction operator in TASK_REDUCTION clause.
!$omp taskgroup task_reduction(.not.: x)
!$omp end taskgroup
end

subroutine f02(p)
  integer, pointer, intent(in) :: p
!ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a TASK_REDUCTION clause
!$omp taskgroup task_reduction(+: p)
!$omp end taskgroup
end

subroutine f03
  common /c/ a, b 
!ERROR: Common block names are not allowed in TASK_REDUCTION clause
!$omp taskgroup task_reduction(+: /c/)
!$omp end taskgroup
end

subroutine f04
  integer :: x(10)
!ERROR: Reference to 'x' must be a contiguous object
!$omp taskgroup task_reduction(+: x(1:10:2))
!$omp end taskgroup
end

subroutine f05
  integer :: x(10)
!ERROR: 'x' in TASK_REDUCTION clause is a zero size array section
!$omp taskgroup task_reduction(+: x(1:0))
!$omp end taskgroup
end

subroutine f06
  type t
    integer :: a(10)
  end type
  type(t) :: x
!ERROR: The base expression of an array element or section in TASK_REDUCTION clause must be an identifier
!$omp taskgroup task_reduction(+: x%a(2))
!$omp end taskgroup
end

subroutine f07
  type t
    integer :: a(10)
  end type
  type(t) :: x
!ERROR: The base expression of an array element or section in TASK_REDUCTION clause must be an identifier
!$omp taskgroup task_reduction(+: x%a(1:10))
!$omp end taskgroup
end

subroutine f08
  integer :: x
!ERROR: Type parameter inquiry is not permitted in TASK_REDUCTION clause
!$omp taskgroup task_reduction(+: x%kind)
!$omp end taskgroup
end
