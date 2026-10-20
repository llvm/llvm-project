!RUN: %python %S/test_errors.py %s %flang_fc1
module m
 contains
  subroutine s(x, y, mask)
    class(*), allocatable, intent(in out) :: x(:), y(:)
    logical, intent(in) :: mask(:)
    select type(x)
    type is(integer)
      print *, 'before, x is integer', x
    type is(real)
      print *, 'before, x is real', x
    class default
      print *, 'before, x has some other type'
    end select
    select type(y)
    type is(integer)
      print *, 'y is integer', y
    type is(real)
      print *, 'y is real', y
    end select
    print *, 'mask', mask
    !ERROR: Assignment to whole polymorphic allocatable 'x' may not be nested in a WHERE statement or construct
    where(mask) x = y
    select type(x)
    type is(integer)
      print *, 'after, x is integer', x
    type is(real)
      print *, 'after, x is real', x
    class default
      print *, 'before, x has some other type'
    end select
    print *
  end
end

program main
  use m
  class(*), allocatable :: x(:), y(:)
  x = [1, 2]
  y = [3., 4.]
  call s(x, y, [.false., .false.])
  x = [1, 2]
  y = [3., 4.]
  call s(x, y, [.false., .true.])
  x = [1, 2]
  y = [3., 4.]
  call s(x, y, [.true., .false.])
  x = [1, 2]
  y = [3., 4.]
  call s(x, y, [.true., .true.])
end program main
