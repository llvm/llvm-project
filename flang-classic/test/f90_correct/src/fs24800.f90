! Test contributed by Neil Carlson from Los Alamos National Laboratory
! related to Flang github issue #243
type foo
  class(*), allocatable :: val
end type
type(foo) :: x
x = foo(42)
end
