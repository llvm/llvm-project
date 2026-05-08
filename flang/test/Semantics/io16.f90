! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
subroutine assumedRank(x, y)
  real x(..), y(*)
  !PORTABILITY: Assumed-rank object 'x' should not be a namelist group item [-Wassumed-rank-io-item]
  !ERROR: A namelist group object 'y' must not be assumed-size
  namelist /nml/x, y
  !ERROR: Assumed-rank object 'x' may not be an I/O list item
  !ERROR: Whole assumed-size array 'y' may not appear here without subscripts
  read *, x, y
  !ERROR: Assumed-rank object 'x' may not be an I/O list item
  !ERROR: Whole assumed-size array 'y' may not appear here without subscripts
  print *, x, y
  read(*,nml=nml)
  write(*,nml=nml)
end
