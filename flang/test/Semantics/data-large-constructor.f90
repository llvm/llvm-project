! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine large_complex
  type pair
    real first
    complex values(999999999)
  end type
  type(pair) x
  !ERROR: not yet implemented: 'x' is too large to initialize with a DATA statement
  data x / pair(-1.1, (1.0, 2.0)) /
end
