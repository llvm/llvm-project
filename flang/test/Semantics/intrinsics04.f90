! RUN: %python %S/test_errors.py %s %flang_fc1 -Wportability
! A potentially absent actual argument cannot require data type conversion.
subroutine s(o,a,p)
  integer(2), intent(in), optional :: o
  integer(2), intent(in), allocatable :: a
  integer(2), intent(in), pointer :: p
  !ERROR: An actual argument to MAX/MIN requiring data conversion may not be OPTIONAL, POINTER, or ALLOCATABLE
  print *, max(1, 2, o)
  !ERROR: An actual argument to MAX/MIN requiring data conversion may not be OPTIONAL, POINTER, or ALLOCATABLE
  print *, max(1, 2, a)
  !ERROR: An actual argument to MAX/MIN requiring data conversion may not be OPTIONAL, POINTER, or ALLOCATABLE
  print *, max(1, 2, p)
  !ERROR: An actual argument to MAX/MIN requiring data conversion may not be OPTIONAL, POINTER, or ALLOCATABLE
  print *, min(1, 2, o)
  !ERROR: An actual argument to MAX/MIN requiring data conversion may not be OPTIONAL, POINTER, or ALLOCATABLE
  print *, min(1, 2, a)
  !ERROR: An actual argument to MAX/MIN requiring data conversion may not be OPTIONAL, POINTER, or ALLOCATABLE
  print *, min(1, 2, p)
  print *, max(1_2, 2_2, o) ! ok
  print *, max(1_2, 2_2, a) ! ok
  print *, max(1_2, 2_2, p) ! ok
  print *, min(1_2, 2_2, o) ! ok
  print *, min(1_2, 2_2, a) ! ok
  print *, min(1_2, 2_2, p) ! ok
end

subroutine ichar_tests()
  integer, parameter :: a1 = ichar('B')
  !WARNING: Character in intrinsic function ichar should have length one [-Wportability]
  integer, parameter :: a2 = ichar('B ')
  !ERROR: Character in intrinsic function ichar must have length one
  !ERROR: Must be a constant value
  integer, parameter :: a3 = ichar('')
end subroutine
