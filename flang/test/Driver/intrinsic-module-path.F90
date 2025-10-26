! Ensure argument -fintrinsic-modules-path works as expected.

!-----------------------------------------
! FLANG DRIVER
!-----------------------------------------
! NOTE: Depending on how Flang is built, the default intrinsics may have higher
!       or lower priority than -fintrinsic-modules-path added here. Using
!       basictestmoduleone.mod from Inputs/module-dir/ will trigger an error.

! RUN:     %flang -fsyntax-only -### %s 2>&1 | FileCheck %s --check-prefix=DEFAULTPATH

! RUN:     %flang -fsyntax-only -DINTRINSICS_DEFAULT %s
! RUN: not %flang -fsyntax-only -DINTRINSICS_INPUTONE %s 2>&1 | FileCheck %s --check-prefix=NOINPUTONE
! RUN: not %flang -fsyntax-only -DINTRINSICS_INPUTTWO %s 2>&1 | FileCheck %s --check-prefix=NOINPUTTWO
! RUN:     %flang -fsyntax-only -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir/ %s
! RUN:     %flang -fsyntax-only -DINTRINSICS_INPUTONE -fintrinsic-modules-path=%S/Inputs/ %s
! RUN:     %flang -fsyntax-only -DINTRINSICS_INPUTONE -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/ -fintrinsic-modules-path=%S/Inputs/module-dir/ %s
! RUN: not %flang -fsyntax-only -DINTRINSICS_INPUTONE -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir/ -fintrinsic-modules-path=%S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=WRONGINPUTONE


!-----------------------------------------
! FLANG FRONTEND (flang -fc1)
!-----------------------------------------
! NOTE: %flang_cc1 the default intrinsics path always has higher priority than
!       -fintrinsic-modules-path added here. Accidentally using
!       ieee_arithmetic/iso_fortran_env from the Inputs/ directory will trigger
!       an error (e.g. when the default intrinsics dir is empty).

! RUN:     %flang_fc1 -fsyntax-only -DINTRINSICS_DEFAULT %s
! RUN: not %flang_fc1 -fsyntax-only -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTONE %s 2>&1 | FileCheck %s --check-prefix=NOINPUTONE
! RUN: not %flang_fc1 -fsyntax-only -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTTWO %s 2>&1 | FileCheck %s --check-prefix=NOINPUTTWO
! RUN:     %flang_fc1 -fsyntax-only -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir %s
! RUN:     %flang_fc1 -fsyntax-only -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTONE -fintrinsic-modules-path=%S/Inputs/ %s
! RUN:     %flang_fc1 -fsyntax-only -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTONE -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/ -fintrinsic-modules-path=%S/Inputs/module-dir/ %s
! RUN: not %flang_fc1 -fsyntax-only -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTONE -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir -fintrinsic-modules-path=%S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=WRONGINPUTONE


! DEFAULTPATH: flang{{.*}}-fc1{{.*}}-fintrinsic-modules-path

! NOINPUTONE: Source file 'basictestmoduleone.mod' was not found
! NOINPUTTWO: Source file 'basictestmoduletwo.mod' was not found
! WRONGINPUTONE: 't1' not found in module 'basictestmoduleone'

program test_intrinsic_module_path
#ifdef INTRINSICS_DEFAULT
   use ieee_arithmetic, only: ieee_round_type
   use iso_fortran_env, only: team_type, event_type, lock_type
#endif
#ifdef INTRINSICS_INPUTONE
   use basictestmoduleone, only: t1
#endif
#ifdef INTRINSICS_INPUTTWO
   use basictestmoduletwo, only: t2
#endif
end program
