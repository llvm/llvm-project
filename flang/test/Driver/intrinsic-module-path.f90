! Ensure argument -fintrinsic-modules-path works as expected.
! WITHOUT the option, the default location for the module is checked and no error generated.
! With the option WRONG and GIVEN, find the module in the first
! -fintrinsic-modules-path that contains a matching file.

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!-----------------------------------------
! RUN: not %flang_fc1 -fsyntax-only                                                                                                     %s 2>&1 | FileCheck %s --check-prefix=WITHOUT
! RUN: not %flang_fc1 -fsyntax-only -fintrinsic-modules-path %S/Inputs/module-dir                                                       %s 2>&1 | FileCheck %s --check-prefix=WRONG
! RUN: not %flang_fc1 -fsyntax-only -fintrinsic-modules-path %S/Inputs/module-dir     -fintrinsic-modules-path %S/Inputs/module-dir-one %s 2>&1 | FileCheck %s --check-prefix=WRONG
! RUN:     %flang_fc1 -fsyntax-only -fintrinsic-modules-path=%S/Inputs/module-dir-one                                                   %s 2>&1 | FileCheck %s --allow-empty --check-prefix=GIVEN
! RUN:     %flang_fc1 -fsyntax-only -fintrinsic-modules-path %S/Inputs/module-dir-one                                                   %s 2>&1 | FileCheck %s --allow-empty --check-prefix=GIVEN
! RUN:     %flang_fc1 -fsyntax-only -fintrinsic-modules-path %S/Inputs/module-dir-one -fintrinsic-modules-path %S/Inputs/module-dir     %s 2>&1 | FileCheck %s --allow-empty --check-prefix=GIVEN

!-----------------------------------------
! BURNSIDE BRIDGE_COMPILER (bbc)
!-----------------------------------------
! RUN: not bbc                                                                                                     %s 2>&1 | FileCheck %s --check-prefix=WITHOUT
! RUN: not bbc -fintrinsic-modules-path %S/Inputs/module-dir                                                       %s 2>&1 | FileCheck %s --check-prefix=WRONG
! RUN: not bbc -fintrinsic-modules-path %S/Inputs/module-dir     -fintrinsic-modules-path %S/Inputs/module-dir-one %s 2>&1 | FileCheck %s --check-prefix=WRONG
! RUN:     bbc -fintrinsic-modules-path=%S/Inputs/module-dir-one                                                   %s 2>&1 | FileCheck %s --allow-empty --check-prefix=GIVEN
! RUN:     bbc -fintrinsic-modules-path %S/Inputs/module-dir-one                                                   %s 2>&1 | FileCheck %s --allow-empty --check-prefix=GIVEN
! RUN:     bbc -fintrinsic-modules-path %S/Inputs/module-dir-one -fintrinsic-modules-path %S/Inputs/module-dir     %s 2>&1 | FileCheck %s --allow-empty --check-prefix=GIVEN


! WITHOUT: error: Cannot parse module file for module 'basictestmoduleone': Source file 'basictestmoduleone.mod' was not found

! WRONG: error: 't1' not found in module 'basictestmoduleone'

! Do not emit any message about the absence of basictestmoduleone
! GIVEN-NOT: basictestmoduleone


program test_intrinsic_modules_path
   use basictestmoduleone, only: t1
end program
