!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00
  integer :: x, y

  ! The x is a direct argument of the + operator. Expect no diagnostics.
  !$omp atomic update
  x = x + (y - 1)
end

subroutine f01
  integer :: x

  ! x + 0 is unusual, but legal. Expect no diagnostics.
  !$omp atomic update
  x = x + 0
end

subroutine f02
  integer :: x

  ! This is formally not allowed by the syntax restrictions of the spec,
  ! but it's equivalent to either x+0 or x*1, both of which are legal.
  ! Allow this case. Expect no diagnostics.
  !$omp atomic update
  x = x
end

subroutine f03
  integer :: x, y

  !$omp atomic update
  !ERROR: The atomic variable x should occur exactly once among the arguments of the top-level + operator
  x = (x + y) + 1
end

subroutine f04
  integer :: x
  real :: y

  !$omp atomic update
  !ERROR: This intrinsic function is not a valid ATOMIC UPDATE operation
  x = floor(x + y)
end

subroutine f05
  integer :: x
  real :: y

  ! An explicit conversion is accepted as an extension.
  !$omp atomic update
  x = int(x + y)
end

subroutine f06
  integer :: x, y
  interface
    function f(i, j)
      integer :: f, i, j
    end
  end interface

  !$omp atomic update
  !ERROR: A call to this function is not a valid ATOMIC UPDATE operation
  x = f(x, y)
end

subroutine f07
  real :: x
  integer :: y

  !$omp atomic update
  !ERROR: The ** operator is not a valid ATOMIC UPDATE operation
  x = x ** y
end

subroutine f08
  integer :: x, y

  !$omp atomic update
  !ERROR: The atomic variable x should appear as an argument in the update operation
  x = y
end
