! Ensure argument -fintrinsic-modules-path works as expected.
! WITHOUT the option, the default location for the module is checked and no error generated.
! With the option GIVEN, the module with the same name is PREPENDED, and considered over the
! default one, causing a CHECKSUM error.

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: %flang_fc1 -fsyntax-only %s  2>&1 | FileCheck %s --allow-empty --check-prefix=WITHOUT
! RUN: not %flang_fc1 -fsyntax-only -fintrinsic-modules-path %S/Inputs/ %s  2>&1 | FileCheck %s --check-prefix=GIVEN

!-----------------------------------------
! EXPECTED OUTPUT WITHOUT
!-----------------------------------------
! WITHOUT-NOT: 'ieee_arithmetic.mod' was not found
! WITHOUT-NOT: 'iso_fortran_env.mod' was not found

!-----------------------------------------
! EXPECTED OUTPUT WITH
!-----------------------------------------
! GIVEN: error: Cannot read module file for module 'ieee_arithmetic': File has invalid checksum
! GIVEN: error: Cannot read module file for module 'iso_fortran_env': File has invalid checksum


program test_intrinsic_module_path
   use ieee_arithmetic, only: ieee_round_type
   use iso_fortran_env, only: team_type, event_type, lock_type
end program
