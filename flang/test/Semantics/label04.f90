
! RUN: %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: branch into loop body from outside
! CHECK: do 10 i = 1, m
! CHECK: the loop branched into
! CHECK: do 20 j = 1, n

subroutine sub00(a,b,n,m)
  real a(n,m)
  real b(n,m)
  if (n .ne. m) then
     goto 50
  end if
  do 10 i = 1, m
     do 20 j = 1, n
50      a(i,j) = b(i,j) + 2.0
20      continue
10      continue
end subroutine sub00
