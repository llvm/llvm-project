! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
!ERROR: No explicit type declared for 'f'
function f()
  implicit none
end

!ERROR: No explicit type declared for 'y'
subroutine s(x, y)
  implicit none
  integer :: x
end

subroutine s2
  implicit none
  block
    !ERROR: No explicit type declared for 'i'
    i = 1
  end block
contains
  subroutine s3
    !ERROR: No explicit type declared for 'j'
    j = 2
  end subroutine
end subroutine

module m1
  implicit none
contains
  subroutine s1
    implicit real (a-h)
    a1 = 1.
    h1 = 1.
    !ERROR: No explicit type declared for 'i1'
    i1 = 1
    !ERROR: No explicit type declared for 'z1'
    z1 = 2.
  contains
    subroutine ss1
      implicit integer(f-j) ! overlap with host scope import is OK
      a2 = 1.
      h2 = 1
      i2 = 1
      !ERROR: No explicit type declared for 'z2'
      z2 = 2.
    contains
      !ERROR: An internal subprogram may not contain an internal subprogram
      subroutine sss1
        implicit none
        !ERROR: No explicit type declared for 'a3'
        a3 = 1.
      end subroutine
    end subroutine
  end subroutine
  subroutine s2
    !ERROR: No explicit type declared for 'b1'
    b1 = 1.
  end subroutine
end module
