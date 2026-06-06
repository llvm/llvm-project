! RUN: not %flang_fc1 -fdebug-unparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: Label '50' was not found
! CHECK: warning: Label '55' is in a construct that should not be used as a branch target here
! CHECK: Label '70' is not a branch target
! CHECK: Control flow use of '70'
! CHECK: error: Label '80' is in a construct that prevents its use as a branch target here
! CHECK: error: Label '90' is in a construct that prevents its use as a branch target here
! CHECK: error: Label '91' is in a construct that prevents its use as a branch target here
! CHECK: error: Label '92' is in a construct that prevents its use as a branch target here
! CHECK: error: Label '30' is in a construct that prevents its use as a branch target here
! CHECK: error: Label '31' is in a construct that prevents its use as a branch target here
! CHECK-NOT: error: Label '32' is in a construct that prevents its use as a branch target here
! CHECK: error: Label '40' is in a construct that prevents its use as a branch target here
! CHECK: error: Label '41' is in a construct that prevents its use as a branch target here
! CHECK-NOT: error: Label '42' is in a construct that prevents its use as a branch target here

subroutine sub00(a,b,n,m)
  real a(n,m)
  real b(n,m)
  if (n .ne. m) then
     goto 50
  end if
6 n = m
end subroutine sub00

subroutine sub01(a,b,n,m)
  real a(n,m)
  real b(n,m)
  if (n .ne. m) then
     goto 55
  else
55   continue
  end if
60 n = m
end subroutine sub01

subroutine sub02(a,b,n,m)
  real a(n,m)
  real b(n,m)
  if (n .ne. m) then
     goto 70
  else
     return
  end if
70 FORMAT (1x,i6)
end subroutine sub02

subroutine sub03(a,n)
  real a(n)
  forall (j=1:n)
80  a(n) = j
  end forall
  go to 80
end subroutine sub03

subroutine sub04(a,n)
  real a(n)
  where (a > 0)
90  a = 1
  elsewhere (a < 0)
91  a = 2
  elsewhere
92  a = 3
  end where
  if (n - 3) 90, 91, 92
end subroutine sub04

subroutine sub05(a)
  real a(..)
  select rank (a)
  rank(1)
31  goto 30
  rank(2)
    goto 32
32  continue
30  continue
  end select
  goto 31
end

subroutine sub06(a)
  class(*) a
  select type (a)
  type is (integer)
41  goto 40
  type is (real)
    goto 42
42  continue
40  continue
  end select
  goto 41
end
