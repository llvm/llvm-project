! RUN: %python %S/test_errors.py %s %flang_fc1

! COMMON block byte sizes and offsets must fit in signed 64-bit integers.

subroutine biggest
  ! 2305843009213693951 * 4 bytes == huge(0_8) - 3, the largest REAL(4)
  ! array that still fits.
  real :: a(2305843009213693951_8)
  common /fits/ a
end subroutine

subroutine one_object
  real :: a(3000000000000000000_8)
  !ERROR: The size of COMMON block /one/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /one/ a
end subroutine

subroutine several_objects
  real :: a(1999999999999999999_8), b(1999999999999999999_8), &
      c(1999999999999999999_8)
  !ERROR: The size of COMMON block /several/ exceeds the maximum supported size of 9223372036854775807 bytes
  common /several/ a, b, c
end subroutine

subroutine blank_common
  real :: a(1999999999999999999_8), b(1999999999999999999_8)
  !ERROR: The size of COMMON block // exceeds the maximum supported size of 9223372036854775807 bytes
  common a, b
end subroutine
