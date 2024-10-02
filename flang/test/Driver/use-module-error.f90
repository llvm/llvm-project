! Ensure that multiple module directories are not allowed

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang -fsyntax-only -J %S/Inputs/ %s 2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: %flang -fsyntax-only -J %S/Inputs/ -J %S/Inputs/ %s  2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: %flang -fsyntax-only -module-dir %S/Inputs/module-dir %s 2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: %flang -fsyntax-only -module-dir %S/Inputs/module-dir -module-dir %S/Inputs/module-dir %s 2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: %flang -fsyntax-only -module-dir %S/Inputs/module-dir -J%S/Inputs/module-dir %s 2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: not %flang -fsyntax-only -J %S/Inputs/module-dir -J %S/Inputs/ %s  2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang -fsyntax-only -J %S/Inputs/module-dir -module-dir %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang -fsyntax-only -module-dir %S/Inputs/module-dir -J%S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: %flang_fc1 -fsyntax-only -J %S/Inputs/ %s 2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: %flang_fc1 -fsyntax-only -J %S/Inputs/ -J %S/Inputs/ %s 2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: %flang_fc1 -fsyntax-only -module-dir %S/Inputs/module-dir %s 2>&1 | FileCheck %s --allow-empty  --check-prefix=SINGLEINCLUDE
! RUN: %flang_fc1 -fsyntax-only -module-dir %S/Inputs/module-dir -module-dir %S/Inputs/module-dir %s 2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: %flang_fc1 -fsyntax-only -module-dir %S/Inputs/module-dir -J%S/Inputs/module-dir %s 2>&1 | FileCheck %s --allow-empty --check-prefix=SINGLEINCLUDE
! RUN: not %flang_fc1 -fsyntax-only -J %S/Inputs/module-dir -J %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang_fc1 -fsyntax-only -J %S/Inputs/module-dir -module-dir %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang_fc1 -fsyntax-only -module-dir %S/Inputs/module-dir -J%S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE

! DOUBLEINCLUDE:error: Only one '-module-dir/-J' directory allowed
! SINGLEINCLUDE-NOT:error: Only one '-module-dir/-J' directory allowed

program too_many_module_dirs
end
