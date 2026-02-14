! Checks that the module file:
!   * is _saved_
!   * is saved in the _directory specified by the user_
! We use `-fsyntax-only` as it stops after the semantic checks (the module file is generated when sema checks are run)

! ------------------------------------------------------------------------------
! At one time, flang accepted -module-dir<value> (note the lack of a separator
! between -module-dir and <value>). This is no longer allowed.
! -module-dir=<value> is also not allowed
!
! RUN: not %flang -fsyntax-only -module-dir%t %s 2>&1 \
! RUN:      | FileCheck %s -check-prefix=JOINED
!
! RUN: not %flang -fsyntax-only -module-dir=%t %s 2>&1 \
! RUN:      | FileCheck %s -check-prefix=JOINED
!
! JOINED: error: unknown argument: '-module-dir{{.+}}'
!
!--------------------------
! -module-dir <value>
!--------------------------
! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -module-dir %t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod
! RUN: cd -
!
!---------------------------
! -J <value>
!---------------------------
! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -J %t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod
! RUN: cd -

!------------------------------
! -J<value>
!------------------------------
! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -J%t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod
! RUN: cd -

module testmodule
  type::t2
  end type
end
