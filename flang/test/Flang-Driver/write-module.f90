! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -module-dir %t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod

! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -J %t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod

! RUN: rm -rf %t && mkdir -p %t/dir-flang
! RUN: cd %t && %flang -fsyntax-only -J%t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.mod && not ls %t/testmodule.mod

module testmodule
  type::t2
  end type
end
