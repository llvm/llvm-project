! Ensure argument -fintrinsic-modules-path works as expected.

! RUN: not bbc %s -fintrinsic-modules-path=%S/Inputs/module-dir/ -fintrinsic-modules-path=%S/Inputs/module-dir-one/ 2>&1 | FileCheck %s --check-prefix=WRONGINPUTONE

! WRONGINPUTONE: 't1' not found in module 'basictestmoduleone'

program test_intrinsic_module_path
   use ieee_arithmetic, only: ieee_round_type
   use iso_fortran_env, only: team_type, event_type, lock_type
   use basictestmoduleone, only: t1
   use basictestmoduletwo, only: t2
end program
