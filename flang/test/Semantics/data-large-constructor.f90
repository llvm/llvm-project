! REQUIRES: system-linux
! UNSUPPORTED: asan, hwasan, msan, tsan
! Limit address space so oversized constructor regressions fail quickly.
! RUN: ulimit -v 1048576
! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine testcomplex
  real, parameter :: one = 1.0
  real, parameter :: two = 2.0
  complex, parameter :: one_two = (one, two)
  type pair
    real first
    complex r_i(999999999)
  end type
  type(pair) vw
  !ERROR: not yet implemented: 'vw' is too large to initialize with a DATA statement
  data vw / pair(-1.1, one_two) /
end

subroutine nested_complex
  type pair
    real first
    complex r_i(999999999)
  end type
  type holder
    integer tag
    type(pair) p
  end type
  type(holder) vw
  !ERROR: not yet implemented: 'vw' is too large to initialize with a DATA statement
  data vw / holder(7, pair(-1.1, (1.0, 2.0))) /
end

subroutine many_components
  type pair
    complex a(62500001)
    complex b(62500001)
  end type
  type(pair) vw
  !ERROR: not yet implemented: 'vw' is too large to initialize with a DATA statement
  data vw / pair((1.0, 2.0), (3.0, 4.0)) /
end

subroutine just_over_limit
  type t
    real values(250000001)
  end type
  type(t) x
  !ERROR: not yet implemented: 'x' is too large to initialize with a DATA statement
  data x / t(1.0) /
end

subroutine mismatched_object
  type pair
    real first
    complex r_i(999999999)
  end type
  integer small
  !ERROR: DATA statement value could not be converted to the type 'INTEGER(4)' of the object 'small'
  data small / pair(-1.1, (1.0, 2.0)) /
end
