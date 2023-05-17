! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine foo
end
subroutine iso_fortran_env
end
subroutine bad1
  !ERROR: 'foo' is not a module
  use foo
end
subroutine ok1
  use, intrinsic :: iso_fortran_env
end
subroutine ok2
  use iso_fortran_env
end
subroutine bad2
  !ERROR: 'iso_fortran_env' is not a module
  use, non_intrinsic :: iso_fortran_env
end
