!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61 -Werror -Wno-experimental-option

subroutine f00(x)
  integer, pointer :: x
  !ERROR: 'attach-modifier' modifier cannot occur multiple times
  !$omp target map(attach(always), attach(never): x)
  !$omp end target
end

subroutine f01(x)
  integer, pointer :: x
  !ERROR: The 'attach-modifier' modifier can only appear on a map-entering construct or on a DECLARE_MAPPER directive
  !$omp target_exit_data map(attach(always): x)
end

subroutine f02(x)
  integer, pointer :: x
  !ERROR: The 'attach-modifier' modifier can only appear on a map-entering construct or on a DECLARE_MAPPER directive
  !$omp target map(attach(never), from: x)
  !$omp end target
end

subroutine f03(x)
  integer :: x
  !ERROR: A list-item that appears in a map clause with the ATTACH modifier must have a base-pointer
  !$omp target map(attach(always), tofrom: x)
  !$omp end target
end

module m
type t
  integer :: z
end type

type u
  type(t), pointer :: y
end type

contains

subroutine f04(n)
  integer :: n
  type(u) :: x(10)

  !Expect no diagonstics
  !$omp target map(attach(always), to: x(n)%y%z)
  !$omp end target
end
end module

subroutine f05(x)
  integer :: x
  !Expect no diagonstics
  !$omp target map(always, to: x)
  !$omp end target
end

subroutine f06(x)
  integer :: x
  x = 0
  !Expect no diagonstics
  !$omp target defaultmap(tofrom: scalar)
  x = x + 1
  !$omp end target
end

subroutine f07(x)
  integer :: x
  !ERROR: 'variable-category' modifier cannot occur multiple times
  !$omp target defaultmap(tofrom: scalar, aggregate)
  !$omp end target
end
