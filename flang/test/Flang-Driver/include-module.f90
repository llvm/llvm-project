! Ensure argument -I works as expected with module files.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang-new -fsyntax-only -I %S/Inputs -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fsyntax-only -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang-new -fsyntax-only -I %S/Inputs -J %S/Inputs/module-dir %s 2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fsyntax-only -J %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang-new -fsyntax-only -I %S/Inputs -module-dir %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fsyntax-only -module-dir %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang-new -fsyntax-only -J %S/Inputs/module-dir -J %S/Inputs/ %s  2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang-new -fsyntax-only -J %S/Inputs/module-dir -module-dir %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang-new -fsyntax-only -module-dir %S/Inputs/module-dir -J%S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: not %flang-new -fc1 -fsyntax-only -I %S/Inputs -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fc1 -fsyntax-only -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang-new -fc1 -fsyntax-only -I %S/Inputs -J %S/Inputs/module-dir %s 2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fc1 -fsyntax-only -J %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang-new -fc1 -fsyntax-only -I %S/Inputs -module-dir %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fc1 -fsyntax-only -module-dir %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang-new -fc1 -fsyntax-only -J %S/Inputs/module-dir -J %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang-new -fc1 -fsyntax-only -J %S/Inputs/module-dir -module-dir %S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE
! RUN: not %flang-new -fc1 -fsyntax-only -module-dir %S/Inputs/module-dir -J%S/Inputs/ %s 2>&1 | FileCheck %s --check-prefix=DOUBLEINCLUDE

!-----------------------------------------
! EXPECTED OUTPUT FOR MISSING MODULE FILE
!-----------------------------------------
! SINGLEINCLUDE:error: Cannot read module file for module 'basictestmoduletwo'
! SINGLEINCLUDE-NOT:error: Cannot read module file for module 'basictestmoduletwo'
! SINGLEINCLUDE-NOT:error: Derived type 't1' not found
! SINGLEINCLUDE:error: Derived type 't2' not found

!-----------------------------------------
! EXPECTED OUTPUT FOR MISSING MODULE FILE
!-----------------------------------------
! DOUBLEINCLUDE:error: Only one '-module-dir/-J' option allowed

!---------------------------------------
! EXPECTED OUTPUT FOR ALL MODULES FOUND
!---------------------------------------
! INCLUDED-NOT:error: Cannot read module file
! INCLUDED-NOT:error: Derived type 't1' not found
! INCLUDED:error: Derived type 't2' not found

program test_dash_I_with_mod_files
    USE basictestmoduleone
    USE basictestmoduletwo
    type(t1) :: x1 ! t1 defined in Inputs/basictestmoduleone.mod
    type(t2) :: x2 ! t2 defined in Inputs/module-dir/basictestmoduleone.mod
end
