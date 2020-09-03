
! RUN: not %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: DO loop doesn't properly nest
! CHECK: DO loop conflicts
! CHECK: Label '30' cannot be found
! CHECK: Label '40' cannot be found
! CHECK: Label '50' doesn't lexically follow DO stmt

subroutine sub00(a,b,n,m)
  real a(n,m)
  real b(n,m)
  do 10 i = 1, m
     do 20 j = 1, n
        a(i,j) = b(i,j) + 2.0
10      continue
20      continue
end subroutine sub00

subroutine sub01(a,b,n,m)
  real a(n,m)
  real b(n,m)
  do 30 i = 1, m
     do 40 j = 1, n
        a(i,j) = b(i,j) + 10.0
35      continue
45      continue
end subroutine sub01

subroutine sub02(a,b,n,m)
  real a(n,m)
  real b(n,m)
50      continue
  do 50 i = 1, m
     do 60 j = 1, n
        a(i,j) = b(i,j) + 20.0
60      continue
end subroutine sub02
