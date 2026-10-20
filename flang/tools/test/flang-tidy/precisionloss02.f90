! RUN: %check_flang_tidy %s bugprone-precision-loss %t
subroutine s
  real(8) :: i
  real(4) :: j
  i = 1.0_8
  j = 2.0
  i = j ! no warning
end subroutine s
