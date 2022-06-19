! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of integer and real MOD and MODULO
module m1
  logical, parameter :: test_mod_i1 = mod(8, 5) == 3
  logical, parameter :: test_mod_i2 = mod(-8, 5) == -3
  logical, parameter :: test_mod_i3 = mod(8, -5) == 3
  logical, parameter :: test_mod_i4 = mod(-8, -5) == -3

  logical, parameter :: test_mod_r1 = mod(3., 2.) == 1.
  logical, parameter :: test_mod_r2 = mod(8., 5.) == 3.
  logical, parameter :: test_mod_r3 = mod(-8., 5.) == -3.
  logical, parameter :: test_mod_r4 = mod(8., -5.) == 3.
  logical, parameter :: test_mod_r5 = mod(-8., -5.) == -3.

  logical, parameter :: test_modulo_i1 = modulo(8, 5) == 3
  logical, parameter :: test_modulo_i2 = modulo(-8, 5) == 2
  logical, parameter :: test_modulo_i3 = modulo(8, -5) == -2
  logical, parameter :: test_modulo_i4 = modulo(-8, -5) == -3

  logical, parameter :: test_modulo_r1 = modulo(8., 5.) == 3.
  logical, parameter :: test_modulo_r2 = modulo(-8., 5.) == 2.
  logical, parameter :: test_modulo_r3 = modulo(8., -5.) == -2.
  logical, parameter :: test_modulo_r4 = modulo(-8., -5.) == -3.
end module
