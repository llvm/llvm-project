! RUN: %python %S/test_errors.py %s %flang_fc1

program bug219453
  real(8) :: fsource
  logical :: mask
  real :: result
  integer(4) :: i4
  integer(8) :: i8

  !ERROR: Actual argument for 'fsource=' has type 'REAL(8)', but 'tsource=' has type 'REAL(4)'
  result = merge(1.0, fsource, mask)
  !ERROR: Actual argument for 'fsource=' has type 'REAL(8)', but 'tsource=' has type 'REAL(4)'
  result = merge(fsource=fsource, tsource=1.0, mask=mask)
  !ERROR: Actual argument for 'j=' has type 'INTEGER(8)', but 'i=' has type 'INTEGER(4)'
  print *, dshiftl(i4, i8, 1)
end program
