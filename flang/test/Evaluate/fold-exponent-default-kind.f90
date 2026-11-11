! RUN: %python %S/test_folding.py %s %flang_fc1 -fdefault-integer-8

module exponent_default_kind
  logical, parameter :: test_value = exponent(1.0) + 1_8 == 2_8
  logical, parameter :: test_kind = kind(exponent(1.0)) == 8
end module
