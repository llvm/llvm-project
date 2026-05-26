! Ensure argument -fintrinsic-modules-path works as expected.
! REQUIRES: x86-registered-target

!-----------------------------------------
! FLANG DRIVER
!-----------------------------------------
! NOTE: Depending on how Flang is built, the default intrinsics may have higher
!       or lower priority than -fintrinsic-modules-path added here. Using
!       basictestmoduleone.mod from Inputs/module-dir/ will trigger an error.

! RUN:     %flang -fsyntax-only --target=x86_64-unknown-linux-gnu -resource-dir %S/Inputs/resource_dir_with_per_target_subdir %s -### 2>&1 | FileCheck %s --check-prefix=DEFAULTPATH

! RUN:     %flang -fsyntax-only --target=x86_64-unknown-linux-gnu -resource-dir %S/Inputs/resource_dir_with_per_target_subdir %s -cpp -DINTRINSICS_DEFAULT
! RUN: not %flang -fsyntax-only --target=x86_64-unknown-linux-gnu -resource-dir %S/Inputs/resource_dir_with_per_target_subdir %s -cpp -DINTRINSICS_INPUTONE 2>&1 | FileCheck %s --check-prefix=NOINPUTONE
! RUN: not %flang -fsyntax-only --target=x86_64-unknown-linux-gnu -resource-dir %S/Inputs/resource_dir_with_per_target_subdir %s -cpp -DINTRINSICS_INPUTTWO 2>&1 | FileCheck %s --check-prefix=NOINPUTTWO
! RUN:     %flang -fsyntax-only --target=x86_64-unknown-linux-gnu -resource-dir %S/Inputs/resource_dir_with_per_target_subdir %s -cpp -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir/
! RUN:     %flang -fsyntax-only --target=x86_64-unknown-linux-gnu -resource-dir %S/Inputs/resource_dir_with_per_target_subdir %s -cpp -DINTRINSICS_INPUTONE -fintrinsic-modules-path=%S/Inputs/module-dir-one/
! RUN:     %flang -fsyntax-only --target=x86_64-unknown-linux-gnu -resource-dir %S/Inputs/resource_dir_with_per_target_subdir %s -cpp -DINTRINSICS_INPUTONE -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir-one/ -fintrinsic-modules-path=%S/Inputs/module-dir/
! RUN: not %flang -fsyntax-only --target=x86_64-unknown-linux-gnu -resource-dir %S/Inputs/resource_dir_with_per_target_subdir %s -cpp -DINTRINSICS_INPUTONE -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir/ -fintrinsic-modules-path=%S/Inputs/module-dir-one/ 2>&1 | FileCheck %s --check-prefix=WRONGINPUTONE


!-----------------------------------------
! FLANG FRONTEND (flang -fc1)
!-----------------------------------------
! NOTE: With %flang_fc1 includes -fintrinsic-modules-path which makes intrinsic
!       modules have higher priority than any -fintrinsic-modules-path added
!       manually. This is in contrast to the driver (%flang) which adds the
!       default intrinsic module dirs to the end of the -fc1 command line.
!       These commands assume the intrinsic modules are found in those default
!       search paths. Accidentally using ieee_arithmetic/iso_fortran_env from
!       the Inputs/ directory will trigger an error (e.g. when the default
!       intrinsics dir is empty).

! RUN:     %flang_fc1 -fsyntax-only -cpp %s -DINTRINSICS_DEFAULT
! RUN: not %flang_fc1 -fsyntax-only -cpp %s -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTONE 2>&1 | FileCheck %s --check-prefix=NOINPUTONE
! RUN: not %flang_fc1 -fsyntax-only -cpp %s -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTTWO 2>&1 | FileCheck %s --check-prefix=NOINPUTTWO
! RUN:     %flang_fc1 -fsyntax-only -cpp %s -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir/
! RUN:     %flang_fc1 -fsyntax-only -cpp %s -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTONE -fintrinsic-modules-path=%S/Inputs/module-dir-one/
! RUN:     %flang_fc1 -fsyntax-only -cpp %s -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTONE -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir-one/ -fintrinsic-modules-path=%S/Inputs/module-dir/
! RUN: not %flang_fc1 -fsyntax-only -cpp %s -DINTRINSICS_DEFAULT -DINTRINSICS_INPUTONE -DINTRINSICS_INPUTTWO -fintrinsic-modules-path=%S/Inputs/module-dir/     -fintrinsic-modules-path=%S/Inputs/module-dir-one/ 2>&1 | FileCheck %s --check-prefix=WRONGINPUTONE


!-----------------------------------------

! DEFAULTPATH:      -fc1
! DEFAULTPATH-SAME: -fintrinsic-modules-path" "{{.*(\\\\|/)}}resource_dir_with_per_target_subdir{{(\\\\|/)}}finclude{{(\\\\|/)}}flang{{(\\\\|/)}}x86_64-unknown-linux-gnu"

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
