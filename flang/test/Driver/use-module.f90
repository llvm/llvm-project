! Checks that module search directories specified with `-J/-module-dir` and `-I` are handled correctly

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang -fsyntax-only -I %S/Inputs -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED --allow-empty
! RUN: %flang -fsyntax-only -I %S/Inputs -J %S/Inputs/module-dir %s 2>&1 | FileCheck %s --check-prefix=INCLUDED --allow-empty
! RUN: %flang -fsyntax-only -I %S/Inputs -module-dir %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED --allow-empty

! RUN: not %flang -fsyntax-only -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=MISSING_MOD2
! RUN: not %flang -fsyntax-only -J %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=MISSING_MOD2
! RUN: not %flang -fsyntax-only -module-dir %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=MISSING_MOD2

! RUN: not %flang -fsyntax-only -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang -fsyntax-only -J %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang -fsyntax-only -module-dir %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE

!-----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!-----------------------------------------
! RUN: %flang_fc1 -fsyntax-only -I %S/Inputs -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED --allow-empty
! RUN: %flang_fc1 -fsyntax-only -I %S/Inputs -J %S/Inputs/module-dir %s 2>&1 | FileCheck %s --check-prefix=INCLUDED --allow-empty
! RUN: %flang_fc1 -fsyntax-only -I %S/Inputs -module-dir %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=INCLUDED --allow-empty

! RUN: not %flang_fc1 -fsyntax-only -I %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=MISSING_MOD2
! RUN: not %flang_fc1 -fsyntax-only -J %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=MISSING_MOD2
! RUN: not %flang_fc1 -fsyntax-only -module-dir %S/Inputs %s  2>&1 | FileCheck %s --check-prefix=MISSING_MOD2

! RUN: not %flang_fc1 -fsyntax-only -I %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang_fc1 -fsyntax-only -J %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE
! RUN: not %flang_fc1 -fsyntax-only -module-dir %S/Inputs/module-dir %s  2>&1 | FileCheck %s --check-prefix=SINGLEINCLUDE

! INCLUDED-NOT: error

! MISSING_MOD2-NOT:error: Cannot read module file for module 'basictestmoduleone''
! MISSING_MOD2-NOT:error: Derived type 't1' not found
! MISSING_MOD2:error: Cannot read module file for module 'basictestmoduletwo'
! MISSING_MOD2:error: Derived type 't2' not found

! SINGLEINCLUDE-NOT:error: Cannot read module file for module 'basictestmoduleone'
! SINGLEINCLUDE:error: Derived type 't1' not found
! SINGLEINCLUDE-NOT:error: Cannot read module file for module 'basictestmoduletwo'
! SINGLEINCLUDE-NOT:error: Derived type 't2' not found


program test_search_dirs_for_mod_files
    USE basictestmoduleone
    USE basictestmoduletwo
    type(t1) :: x1 ! t1 defined in Inputs/basictestmoduleone.mod
    type(t2) :: x2 ! t2 defined in Inputs/module-dir/basictestmoduleone.mod
end
