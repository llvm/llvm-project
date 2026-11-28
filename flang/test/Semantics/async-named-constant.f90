! RUN: %python %S/test_errors.py %s %flang_fc1
! A named constant that is a data list item of an asynchronous data transfer
! statement is not a variable; it must not implicitly acquire the
! ASYNCHRONOUS attribute.
module m
  integer, parameter :: mc(3) = [4, 5, 6]
end module

program p
  use m
  implicit none
  integer, parameter :: arr(3) = [1, 2, 3]
  integer :: id, v(3)
  open(10, status='scratch', form='unformatted', asynchronous='yes')
  write(10, id=id, asynchronous='yes') arr
  wait(10, id=id)
  write(10, id=id, asynchronous='yes') mc
  wait(10, id=id)
  call internal
  block
    write(10, id=id, asynchronous='yes') arr
    wait(10, id=id)
  end block
  rewind(10)
  read(10) v
 contains
  subroutine internal
    write(10, id=id, asynchronous='yes') arr
    wait(10, id=id)
  end subroutine
end program
