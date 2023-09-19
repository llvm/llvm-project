! RUN: %python %S/test_errors.py %s %flang_fc1 -funderscoring

subroutine conflict()
end subroutine

!ERROR: BIND(C) procedure assembly name conflicts with non BIND(C) procedure assembly name
subroutine foo(x)  bind(c, name="conflict_")
  real :: x
end subroutine
