! RUN: %python %S/test_errors.py %s %flang_fc1 -funsigned -Werror
! Test the warning for a BOZ literal constant that is too large for the
! INTEGER or UNSIGNED type of an assignment target.
program boztest
  integer(4) :: i4
  integer(8) :: i8
  unsigned(2) :: u2
  unsigned(4) :: u4

  ! INTEGER targets
  i4 = z'FFFFFFFF' ! fits in 32 bits, no warning
  i4 = z'0000000FFFFFFFF' ! long BOZ string with non-overflowing value
  !WARNING: BOZ literal constant is too large for INTEGER(KIND=4) assignment target; truncated [-Wboz-literal-truncation]
  i4 = z'1FFFFFFFF' ! 33 bits, too large for INTEGER(4)
  i8 = z'FFFFFFFFFFFFFFFF' ! fits in 64 bits, no warning
  !WARNING: BOZ literal constant is too large for INTEGER(KIND=8) assignment target; truncated [-Wboz-literal-truncation]
  i8 = z'1FFFFFFFFFFFFFFFF' ! 65 bits, too large for INTEGER(8)

  ! UNSIGNED targets
  u2 = z'FFFF' ! fits in 16 bits, no warning
  !WARNING: BOZ literal constant is too large for UNSIGNED(KIND=2) assignment target; truncated [-Wboz-literal-truncation]
  u2 = z'1FFFF' ! 17 bits, too large for UNSIGNED(2)
  u4 = z'FFFFFFFF' ! fits in 32 bits, no warning
  !WARNING: BOZ literal constant is too large for UNSIGNED(KIND=4) assignment target; truncated [-Wboz-literal-truncation]
  u4 = z'1FFFFFFFF' ! 33 bits, too large for UNSIGNED(4)
end program
