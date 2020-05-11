! RUN: %S/test_errors.sh %s %t %f18
  !ERROR: String edit descriptor in READ format expression
  read(*,'("abc")')

  !ERROR: String edit descriptor in READ format expression
  !ERROR: Unterminated format expression
  read(*,'("abc)')

  !ERROR: 'H' edit descriptor in READ format expression
  read(*,'(3Habc)')

  !ERROR: 'H' edit descriptor in READ format expression
  !ERROR: Unterminated format expression
  read(*,'(5Habc)')

  !ERROR: 'I' edit descriptor 'w' value must be positive
  read(*,'(I0)')
end
