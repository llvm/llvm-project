! Checks that the module file:
!   * is _saved_
!   * is saved in the _directory specified by the user_
! We use `-fsyntax-only` as it stops after the semantic checks (the module file is generated when sema checks are run)

!--------------------------
! -module-dir
!--------------------------
! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -module-dir %t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod

!---------------------------
! -J <dir> (i.e. with space)
!---------------------------
! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -J %t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod

!------------------------------
! -J<dir> (i.e. without space)
!------------------------------
! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -J%t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod

module testmodule
  type::t2
  end type
end
