
! RUN: not %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: Label '30' is not a branch target
! CHECK: Control flow use of '30'
! CHECK: Label '10' is not in scope
! CHECK: Label '20' was not found
! CHECK: Label '60' was not found

subroutine sub00(n,m)
30 format (i6,f6.2)
  if (n .eq. m) then
10   print *,"equal"
  end if
  call sub01(n,*10,*20,*30)
  write (*,60) n, m
end subroutine sub00
