! Ensure argument -I works as expected with module files.
! The module files for this test are not real module files.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: not %flang-new -fsyntax-only -I %S/Inputs -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fsyntax-only -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: not %flang-new -fc1 -fsyntax-only -I %S/Inputs -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED
! RUN: not %flang-new -fc1 -fsyntax-only -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE

!-----------------------------------------
! EXPECTED OUTPUT FOR MISSING MODULE FILE
!-----------------------------------------
! SINGLEINCLUDE:No such file or directory
! SINGLEINCLUDE-NOT:Error reading module file for module 'basictestmoduletwo'

!---------------------------------------
! EXPECTED OUTPUT FOR ALL MODULES FOUND
!---------------------------------------
! INCLUDED-NOT:No such file or directory

program test_dash_I_with_mod_files
    USE basictestmoduleone
    USE basictestmoduletwo
end
