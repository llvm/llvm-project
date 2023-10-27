! RUN: %python %S/test_errors.py %s %flang_fc1
program testcriticalconstruct

  Start_name_only: Critical
  !ERROR: CRITICAL construct name required but missing
  End critical !C1117 in the Fortran 2018 standard

  critical
  !ERROR: CRITICAL construct name unexpected
  end critical end_name_only !C1117 in Fortran 2018 standard

end program
