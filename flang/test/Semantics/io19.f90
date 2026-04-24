! RUN: %python %S/test_errors.py %s %flang_fc1
! Test AT edit descriptor (Fortran 2023)

  character(10) :: str
  character(kind=2,len=10) :: str2
  character(kind=4,len=10) :: str4

  ! Valid: AT with WRITE
  write(*, '(AT)') 'hello'
  write(*, '(AT)') str
  write(*, '(2AT)') 'abc', 'def'

  ! Valid: AT with non-default character kinds
  write(*, '(AT)') str2
  write(*, '(AT)') str4
  write(*, '(2AT)') str2, str4

  ! Valid: AT in FORMAT statement for WRITE
1 format(AT)
  write(*,1) 'hello'

  ! Error: AT must not be used for input
  !ERROR: 'AT' edit descriptor must not be used for input
  read(*, '(AT)') str

  ! Error: AT must not be used for input with non-default kinds
  !ERROR: 'AT' edit descriptor must not be used for input
  read(*, '(AT)') str2
  !ERROR: 'AT' edit descriptor must not be used for input
  read(*, '(AT)') str4

  ! AT does not accept a width
  !ERROR: 'AT' edit descriptor does not accept a width value
  write(*, '(AT10)') str
  !ERROR: 'AT' edit descriptor does not accept a width value
  !ERROR: Unexpected '.' in format expression
  write(*, '(AT10.2)') str

  ! AT with width on input produces two independent errors.
  !ERROR: 'AT' edit descriptor does not accept a width value
  !ERROR: 'AT' edit descriptor must not be used for input
  read(*, '(AT10)') str

  ! FORMAT statements are standalone; the compiler cannot know if they will
  ! be used with READ or WRITE, so no compile-time error is expected here.
2 format(AT)
  read(*,2) str
end
