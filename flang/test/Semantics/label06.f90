
! RUN: not %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: Label '10' is not in scope
! CHECK: Label '20' was not found
! CHECK: Label '30' is not a branch target
! CHECK: Control flow use of '30'
! CHECK: Label '40' is not in scope
! CHECK: Label '50' is not in scope

subroutine sub00(n)
  GOTO (10,20,30) n
  if (n .eq. 1) then
10   print *, "xyz"
  end if
30 FORMAT (1x,i6)
end subroutine sub00

subroutine sub01(n)
  real n
  GOTO (40,50,60) n
  if (n .eq. 1) then
40   print *, "xyz"
50 end if
60 continue
end subroutine sub01
