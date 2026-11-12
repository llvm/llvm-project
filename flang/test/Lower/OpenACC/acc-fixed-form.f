! RUN: bbc -ffixed-form -fopenacc -emit-hlfir %s -o - | FileCheck %s

      subroutine sub1()
      real :: a(10, 10)
      integer :: i, j

      a = 0.0

c$acc parallel
      do j = 1, 10
        do i = 1, 10
          a(i,j) = i*j
        end do
      end do
c$acc end parallel

*$acc parallel
      do j = 1, 10
        do i = 1, 10
          a(i,j) = i*j
        end do
      end do
*$acc end parallel

!$acc parallel
      do j = 1, 10
        do i = 1, 10
          a(i,j) = i*j
        end do
      end do
!$acc end parallel

      end subroutine

! CHECK-COUNT-3: acc.parallel
