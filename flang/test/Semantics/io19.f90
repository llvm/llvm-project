! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Test AT edit descriptor (Fortran 202X)

  character(10) :: str

  ! Valid: AT with WRITE
  write(*, '(AT)') 'hello'
  write(*, '(AT)') str
  write(*, '(2AT)') 'abc', 'def'

  ! Valid: AT in FORMAT statement for WRITE
1 format(AT)
  write(*,1) 'hello'

  ! Error: AT must not be used for input
  !ERROR: 'AT' edit descriptor must not be used for input
  read(*, '(AT)') str

  ! AT does not accept a width
  !ERROR: 'AT' edit descriptor does not accept a width value
  write(*, '(AT10)') str

  ! FORMAT statements are standalone; the compiler cannot know if they will
  ! be used with READ or WRITE, so no compile-time error is expected here.
2 format(AT)
  read(*,2) str
end
