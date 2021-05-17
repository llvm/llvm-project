! RUN: %S/test_errors.sh %s %t %flang_fc1 -pedantic

  write(*, '(B0)')
  write(*, '(B3)')

  !WARNING: Expected 'B' edit descriptor 'w' value
  write(*, '(B)')

  !WARNING: Expected 'EN' edit descriptor 'w' value
  !WARNING: Non-standard '$' edit descriptor
  write(*, '(EN,$)')

  !WARNING: Expected 'G' edit descriptor 'w' value
  write(*, '(3G)')

  !WARNING: Non-standard '\' edit descriptor
  write(*,'(A, \)') 'Hello'

  !WARNING: 'X' edit descriptor must have a positive position value
  write(*, '(X)')

  !WARNING: Legacy 'H' edit descriptor
  write(*, '(3Habc)')

  !WARNING: 'X' edit descriptor must have a positive position value
  !WARNING: Expected ',' or ')' in format expression
  !WARNING: 'X' edit descriptor must have a positive position value
  write(*,'(XX)')

  !WARNING: Expected ',' or ')' in format expression
  write(*,'(RZEN8.2)')

  !WARNING: Expected ',' or ')' in format expression
  write(*,'(3P7I2)')

  !WARNING: Expected ',' or ')' in format expression
  write(*,'(5X i3)')
end
