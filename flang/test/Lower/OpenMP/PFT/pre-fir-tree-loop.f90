! RUN: bbc -fopenmp -pft-test -o %t %s | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fdebug-dump-pft -o %t %s | FileCheck %s

! Loop constructs always have an `end do` which can be the target of
! a branch. So OpenMP loop constructs do not need an artificial
! continue inserted for a target.

!CHECK-LABEL: sb0
!CHECK-NOT: continue
subroutine sb0(cond)
  implicit none
  logical :: cond
  integer :: i
  !$omp parallel do
  do i = 1, 20
    if( cond) then
      cycle
    end if
  end do
  return
end subroutine

!CHECK-LABEL: sb1
!CHECK-NOT: continue
subroutine sb1(cond)
  implicit none
  logical :: cond
  integer :: i
  !$omp parallel do
  do i = 1, 20
    if( cond) then
      cycle
    end if
  end do
  !$omp end parallel do
  return
end subroutine

!CHECK-LABEL: sb2
!CHECK-NOT: continue
subroutine sb2
  integer :: i, n
  integer :: tmp

  !$omp parallel do
  do ifld=1,n
     do isum=1,n
       if (tmp > n) then
         exit
       endif
     enddo
     tmp = n
  enddo
end subroutine

!CHECK-LABEL: sb3
!CHECK-NOT: continue
subroutine sb3
  integer :: i, n
  integer :: tmp

  !$omp parallel do
  do ifld=1,n
     do isum=1,n
       if (tmp > n) then
         exit
       endif
     enddo
  enddo
end subroutine
