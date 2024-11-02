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

  logical, parameter :: test_mod_r10a = mod(huge(0.), tiny(0.)) == 0.
  logical, parameter :: test_mod_r10b = sign(1., mod(huge(0.), tiny(0.))) == 1.
  logical, parameter :: test_mod_r11a = mod(-huge(0.), tiny(0.)) == 0.
  logical, parameter :: test_mod_r11b = sign(1., mod(-huge(0.), tiny(0.))) == -1.
  logical, parameter :: test_mod_r12a = mod(huge(0.), -tiny(0.)) == 0.
  logical, parameter :: test_mod_r12b = sign(1., mod(huge(0.), -tiny(0.))) == 1.
  logical, parameter :: test_mod_r13a = mod(huge(0.), tiny(0.)) == 0.
  logical, parameter :: test_mod_r13b = sign(1., mod(-huge(0.), -tiny(0.))) == -1.

  logical, parameter :: test_modulo_i1 = modulo(8, 5) == 3
  logical, parameter :: test_modulo_i2 = modulo(-8, 5) == 2
  logical, parameter :: test_modulo_i3 = modulo(8, -5) == -2
  logical, parameter :: test_modulo_i4 = modulo(-8, -5) == -3

  logical, parameter :: test_modulo_r1 = modulo(8., 5.) == 3.
  logical, parameter :: test_modulo_r2 = modulo(-8., 5.) == 2.
  logical, parameter :: test_modulo_r3 = modulo(8., -5.) == -2.
  logical, parameter :: test_modulo_r4 = modulo(-8., -5.) == -3.

  logical, parameter :: test_modulo_r10a = modulo(huge(0.), tiny(0.)) == 0.
  logical, parameter :: test_modulo_r10b = sign(1., modulo(huge(0.), tiny(0.))) == 1.
  logical, parameter :: test_modulo_r11a = modulo(-huge(0.), tiny(0.)) == 0.
  logical, parameter :: test_modulo_r11b = sign(1., modulo(-huge(0.), tiny(0.))) == 1.
  logical, parameter :: test_modulo_r12a = modulo(huge(0.), -tiny(0.)) == 0.
  logical, parameter :: test_modulo_r12b = sign(1., modulo(huge(0.), -tiny(0.))) == -1.
  logical, parameter :: test_modulo_r13a = modulo(huge(0.), tiny(0.)) == 0.
  logical, parameter :: test_modulo_r13b = sign(1., modulo(-huge(0.), -tiny(0.))) == -1.
end module
