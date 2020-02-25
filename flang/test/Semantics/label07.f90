! negative test -- invalid labels, out of range

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: '30' not a branch target
! CHECK: control flow use of '30'
! CHECK: label '10' is not in scope
! CHECK: label '20' was not found
! CHECK: label '60' was not found

subroutine sub00(n,m)
30 format (i6,f6.2)
  if (n .eq. m) then
10   print *,"equal"
  end if
  call sub01(n,*10,*20,*30)
  write (*,60) n, m
end subroutine sub00
