! Tests `-module-suffix` frontend option

!--------------------------
! RUN lines
!--------------------------
! RUN: rm -rf %t && mkdir -p %t/dir-flang/
! RUN: cd %t && %flang_fc1 -fsyntax-only -module-suffix .f18.mod -module-dir %t/dir-flang %s
! RUN: ls %t/dir-flang/testmodule.f18.mod && not ls %t/dir-flang/testmodule.mod

!--------------------------
! INPUT
!--------------------------
module testmodule
  type::t2
  end type
end
