! RUN: %flang_fc1 -fdebug-unparse-with-symbols -DSTRICT_F18 -pedantic %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-unparse-with-symbols -DARCHAIC_FORTRAN %s 2>&1 | FileCheck %s
! CHECK-NOT: :{{[[:space:]]}}error:{{[[:space:]]}}
! FIXME: the above check line does not work because diags are not emitted with error: in them.

! these are the conformance tests
! define STRICT_F18 to eliminate tests of features not in F18
! define ARCHAIC_FORTRAN to add test of feature found in Fortran before F95

subroutine sub00(a,b,n,m)
  integer :: n, m
  real a(n)
  real :: b(m)
1 print *, n, m
1234 print *, a(n), b(1)
99999 print *, a(1), b(m)
end subroutine sub00

subroutine do_loop01(a,n)
  integer :: n
  real, dimension(n) :: a
  do 10 i = 1, n
     print *, i, a(i)
10   continue
end subroutine do_loop01

subroutine do_loop02(a,n)
  integer :: n
  real, dimension(n,n) :: a
  do 10 j = 1, n
     do 10 i = 1, n
        print *, i, j, a(i, j)
10      continue
end subroutine do_loop02

#ifndef STRICT_F18
subroutine do_loop03(a,n)
  integer :: n
  real, dimension(n) :: a
  do 10 i = 1, n
10   print *, i, a(i)		! extension (not f18)
end subroutine do_loop03

subroutine do_loop04(a,n)
  integer :: n
  real :: a(n,n)
  do 10 j = 1, n
     do 10 i = 1, n
10      print *, i, j, a(i, j)	! extension (not f18)
end subroutine do_loop04

subroutine do_loop05(a,n)
  integer :: n
  real a(n,n,n)
  do 10 k = 1, n
     do 10 j = 1, n
        do 10 i = 1, n
10         print *, a(i, j, k)	! extension (not f18)
end subroutine do_loop05
#endif

subroutine do_loop06(a,n)
  integer :: n
  real, dimension(n) :: a
  loopname: do i = 1, n
     print *, i, a(i)
     if (i .gt. 50) then
678     exit
     end if
  end do loopname
end subroutine do_loop06

subroutine do_loop07(a,n)
  integer :: n
  real, dimension(n,n) :: a
  loopone: do j = 1, n
     looptwo: do i = 1, n
        print *, i, j, a(i, j)
     end do looptwo
  end do loopone
end subroutine do_loop07

#ifndef STRICT_F18
subroutine do_loop08(a,b,n,m,nn)
  integer :: n, m, nn
  real, dimension(n,n) :: a
  real b(m,nn)
  loopone: do j = 1, n
     condone: if (m .lt. n) then
        looptwo: do i = 1, m
           condtwo: if (n .lt. nn) then
              b(m-i,j) = s(m-i,j)
              if (i .eq. j) then
                 goto 111
              end if
           else
              cycle loopone
           end if condtwo
        end do looptwo
     else if (n .lt. m) then
        loopthree: do i = 1, n
           condthree: if (n .lt. nn) then
              a(i,j) = b(i,j)
              if (i .eq. j) then
                 return
              end if
           else
              exit loopthree
           end if condthree
        end do loopthree
     end if condone
  end do loopone
111 print *, "done"
end subroutine do_loop08
#endif

#ifndef STRICT_F18
! extended ranges supported by PGI, gfortran gives warnings
subroutine do_loop09(a,n,j)
  integer :: n
  real a(n)
  goto 400
200 print *, "found the index", j
  print *, "value at", j, "is", a(j)
  goto 300 ! FIXME: emits diagnostic even without -pedantic
400  do 100 i = 1, n
     if (i .eq. j) then
        goto 200	! extension: extended GOTO ranges
300     continue
     else
        print *, a(i)
     end if
100 end do
500 continue
end subroutine do_loop09
#endif

subroutine goto10(a,b,n)
  dimension :: a(3), b(3)
  goto 10
10 print *,"x"
4 labelit: if (a(n-1) .ne. b(n-2)) then
     goto 567
  end if labelit
567 end subroutine goto10

subroutine computed_goto11(i,j,k)
  goto (100,110,120) i
100 print *, j
  goto 200
110 print *, k
  goto 200
120 print *, -1
200 end subroutine computed_goto11

#ifndef STRICT_F18
subroutine arith_if12(i)
  if (i) 300,310,320
300 continue
  print *,"<"
  goto 340
310 print *,"=="
340 goto 330
320 print *,">"
330 goto 350
350 continue
end subroutine arith_if12
#endif

#ifndef STRICT_F18
subroutine alt_return_spec13(i,*,*,*)
9 continue
8 labelme: if (i .lt. 42) then
7  return 1
6 else if (i .lt. 94) then
5  return 2
4 else if (i .lt. 645) then
3  return 3
2 end if labelme
1 end subroutine alt_return_spec13

subroutine alt_return_spec14(i)
  call alt_return_spec13(i,*6000,*6130,*6457)
  print *, "Hi!"
6000 continue
6100 print *,"123"
6130 continue
6400 print *,"abc"
6457 continue
6650 print *,"!@#"
end subroutine alt_return_spec14
#endif

#ifndef STRICT_F18
subroutine specifiers15(a,b,x)
  integer x
  OPEN (10, file="myfile.dat", err=100)
  READ (10,20,end=200,size=x,advance='no',eor=300) a
  goto 99
99 CLOSE (10)
  goto 40
100 print *,"error opening"
101 return
200 print *,"end of file"
202 return
300 print *, "end of record"
303 return
20 FORMAT (1x,F5.1)
30 FORMAT (2x,F6.2)
40 OPEN (11, file="myfile2.dat", err=100)
  goto 50
50 WRITE (11,30,err=100) b
  CLOSE (11)
end subroutine specifiers15
#endif

#if !defined(STRICT_F18) && defined(ARCHAIC_FORTRAN)
! assigned goto was deleted in F95. PGI supports, gfortran gives warnings
subroutine assigned_goto16
  assign 10 to i
  goto i (10, 20, 30)
10 continue
  assign 20 to i
20 continue
  assign 30 to i
30 pause
  print *, "archaic feature!"
end subroutine assigned_goto16
#endif
