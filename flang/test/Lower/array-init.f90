! RUN: bbc -emit-llvm -o - %s | tco | llc | as -o %t
! RUN: cc %t %S/array-init-driver.c
! RUN: ./a.out | FileCheck %s

subroutine setall(a, x)
  real :: a(10,20)
  real :: x
  integer :: i, j
  do i = 2, 9
     a(i, 1) = -1.0
     do j = 2, 19
        a(i, j) = x
     end do
     a(i, 20) = -2.0
  end do
  do j = 1, 20
     a(1, j) = 0.0
     a(10, j) = -3.0
  end do
end subroutine setall

! Two subroutines that mean the same thing semantically. sub1 has explicit
! loops over the arrays, but there are no loop-carried dependences. The
! operation can be performed concurrently across the entire 2-D iteration
! space. sub2 has implicit loops and obviates the need for any analyis.

! expected results from our temporary .c driver
!
! CHECK-LABEL: sub1
! CHECK: c(1,1) = 0.0
! CHECK: c(2,9) = 9.0
! CHECK: c(6,6) = 7.0

subroutine sub1(a,b,c)
  real :: a(10,20), b(10,20), c(2:11,20)
  integer :: i, j
  do i = 1, 10
     do j = 1, 20
        a(i,j) = b(i,j) + c(i+1,j)
     end do
  end do
end subroutine sub1

!subroutine sub2(a,b,c)
!  real :: a(10,20), b(10,20), c(10,20)
!  integer :: i, j
!  a = b + c
!end subroutine sub2
