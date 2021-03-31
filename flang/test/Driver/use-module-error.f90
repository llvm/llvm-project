! Ensure that multiple module directories are not allowed

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang -fsyntax-only -J %S/Inputs/module-dir -J %S/Inputs/ %s  2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang -fsyntax-only -J %S/Inputs/module-dir -module-dir %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang -fsyntax-only -module-dir %S/Inputs/module-dir -J%S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: not %flang_fc1 -fsyntax-only -J %S/Inputs/module-dir -J %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang_fc1 -fsyntax-only -J %S/Inputs/module-dir -module-dir %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang_fc1 -fsyntax-only -module-dir %S/Inputs/module-dir -J%S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE

!-----------------------------------------
! EXPECTED OUTPUT FOR MISSING MODULE FILE
!-----------------------------------------
! DOUBLEINCLUDE:error: Only one '-module-dir/-J' option allowed

program too_many_module_dirs
end
