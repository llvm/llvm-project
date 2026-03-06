! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

module m
  implicit none
  integer, parameter :: mc = 42
end module

! Local named constant
program p
  use m
  implicit none
  integer, parameter :: k = 3
  !PORTABILITY: A namelist group object 'k' should not be a PARAMETER [-Wnamelist-parameter]
  namelist /g/ k
  ! USE-associated named constant
  !PORTABILITY: A namelist group object 'mc' should not be a PARAMETER [-Wnamelist-parameter]
  namelist /g2/ mc
end program

! Host-associated named constant
subroutine host
  implicit none
  integer, parameter :: hc = 10
  contains
    subroutine inner
      !PORTABILITY: A namelist group object 'hc' should not be a PARAMETER [-Wnamelist-parameter]
      namelist /g3/ hc
    end subroutine
end subroutine
