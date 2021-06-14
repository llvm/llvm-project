! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Regression test for subscript error recovery
module m
  implicit none
  integer, parameter :: n = 3
  integer, parameter :: pc(n) = [0, 5, 6]
 contains
  logical function f(u)
    integer :: u
    !ERROR: No explicit type declared for 'i'
    do i = 1, n
      !ERROR: No explicit type declared for 'i'
      if (pc(i) == u) then
        f = .true.
        return
      end if
    end do
    f = .false.
  end
end module
