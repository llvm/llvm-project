!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00
  integer :: x, v
  ! The end-directive is optional in ATOMIC WRITE. Expect no diagnostics.
  !$omp atomic write
  x = v + 1

  !$omp atomic write
  x = v + 3
  !$omp end atomic
end

subroutine f01
  integer, pointer :: x, v
  ! Intrinsic assignment and pointer assignment are both ok. Expect no
  ! diagnostics.
  !$omp atomic write
  x = 2 * v + 3

  !$omp atomic write
  x => v
end

subroutine f02(i)
  integer :: i, v
  interface
    function p(i)
      integer, pointer :: p
      integer :: i
    end
  end interface

  ! Atomic variable can be a function reference. Expect no diagostics.
  !$omp atomic write
  p(i) = v
end

subroutine f03
  integer :: x(3), y(5), v(3)

  !$omp atomic write
  !ERROR: Atomic variable x should be a scalar
  x = v

  !$omp atomic write
  !ERROR: Atomic variable y(2_8:4_8:1_8) should be a scalar
  y(2:4) = v
end

subroutine f04
  integer :: x, y(3), v

  !$omp atomic write
  !ERROR: Within atomic operation x and x+1_4 access the same storage
  x = x + 1

  ! Accessing same array, but not the same storage. Expect no diagnostics.
  !$omp atomic write
  y(1) = y(2)
end

subroutine f06
  character :: x, v

  !$omp atomic write
  !ERROR: Atomic variable x cannot have CHARACTER type
  x = v
end

subroutine f07
  integer, allocatable :: x
  integer :: v

  allocate(x)

  !$omp atomic write
  !ERROR: Atomic variable x cannot be ALLOCATABLE
  x = v
end

