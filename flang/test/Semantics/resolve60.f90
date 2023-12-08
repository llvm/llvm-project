! RUN: %python %S/test_errors.py %s %flang_fc1
! Testing 7.6 enum

  ! OK
  enum, bind(C)
    enumerator :: red, green
    enumerator blue, pink
    enumerator yellow
    enumerator :: purple = 2
  end enum

  integer(yellow) anint4

  enum, bind(C)
    enumerator :: square, cicrle
    !ERROR: 'square' is already declared in this scoping unit
    enumerator square
  end enum

  dimension :: apple(4)
  real :: peach

  enum, bind(C)
    !ERROR: 'apple' is already declared in this scoping unit
    enumerator :: apple
    enumerator :: pear
    !ERROR: 'peach' is already declared in this scoping unit
    enumerator :: peach
    !ERROR: 'red' is already declared in this scoping unit
    enumerator :: red
  end enum

  enum, bind(C)
    !ERROR: Enumerator value could not be computed from the given expression
    !WARNING: INTEGER(4) division by zero
    !ERROR: Must be a constant value
    enumerator :: wrong = 0/0
  end enum

end
