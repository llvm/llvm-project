! RUN: %python %S/test_errors.py %s %flang_fc1

program bug219453
  real(8) :: fsource
  logical :: mask
  real :: result

  !ERROR: Actual argument for 'fsource=' has type 'REAL(8)', but 'tsource=' has type 'REAL(4)'
  result = merge(1.0, fsource, mask)
end program
