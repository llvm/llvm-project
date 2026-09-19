! RUN: %check_flang_tidy %s bugprone-precision-loss %t
subroutine precision_test
  integer :: i
  real :: r
  real(8) :: d
  complex :: c
  complex(8) :: z

  ! Integer to real - no precision loss
  r = i  ! No warning

  ! Real to double - no precision loss
  d = r  ! No warning

  ! Double to real - precision loss
  r = d
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Possible loss of precision in implicit conversion (REAL(8) to REAL(4))

  ! Complex to real
  r = c

  ! Complex(8) to complex(4) - precision loss
  c = z
  ! CHECK-MESSAGES: :[[@LINE-1]]:3: warning: Possible loss of precision in implicit conversion (COMPLEX(8) to COMPLEX(4))

  ! Real to complex - no precision loss
  c = r  ! No warning

  ! Literal with kind - no precision loss
  d = 1.0_8  ! No warning
end subroutine precision_test
