!RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  type pair
  end type
  interface pair
    module procedure f
  end interface
 contains
  type(pair) function f(n)
    integer, intent(in) :: n
    f = pair()
  end
end
module m2
  type pair
  end type
end
module m3
  type pair
  end type
end
program main
  use m1
  use m2
  use m3
  !ERROR: Reference to 'pair' is ambiguous
  type(pair) error
end

module m4
  type pair_subroutine
  end type
  !WARNING: Generic interface 'pair_subroutine' should only contain functions due to derived type with same name [-Wsubroutine-and-function-specifics]
  interface pair_subroutine
    subroutine s(var)
    end subroutine
  end interface
end

module m5
  type pair_subroutine
  end type
end

! The generic remains usable after its homonymous derived type becomes
! ambiguous due to USE association.
subroutine test_generic_is_preserved
  use m4
  use m5
  call pair_subroutine(1.)
end

! An unused ambiguous derived type nested in a generic must not cause a crash
! while the generic's specific procedures are checked.
subroutine test_unused_ambiguous_derived_type
  use m4
  use m5
end

subroutine test_ambiguous_derived_type
  use m4
  use m5
  !ERROR: Reference to 'pair_subroutine' is ambiguous
  type(pair_subroutine) error
end

subroutine test_ambiguous_derived_type_reverse_use_order
  use m5
  use m4
  !ERROR: Reference to 'pair_subroutine' is ambiguous
  type(pair_subroutine) error
end
