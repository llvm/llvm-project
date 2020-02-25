! negative test -- invalid labels, out of range

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: label '50' was not found
! CHECK: label '55' is not in scope
! CHECK: '70' not a branch target
! CHECK: control flow use of '70'

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
