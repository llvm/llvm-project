! Test contributed by Neil Carlson from Los Alamos National Laboratory
! related to Flang github issue #243

type foo
  class(*), allocatable :: val
end type
type(foo) :: x
x = foo(42)
select type (val => x%val)
type is (integer)
  if (val /= 42) then
    print *, "FAIL 1"
  else
    print *, "PASS"
  end if
class default
  print *, "FAIL 2"
end select
end
