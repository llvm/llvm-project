
! RUN: bbc -fopenacc -fcuda -emit-hlfir %s -o - | FileCheck %s

module m

interface doit
subroutine __device_sub(a)
    real(4), device, intent(in) :: a(:,:,:)
    !dir$ ignore_tkr(c) a
end
subroutine __host_sub(a)
    real(4), intent(in) :: a(:,:,:)
    !dir$ ignore_tkr(c) a
end
end interface
end module

program testex1
integer, parameter :: ntimes = 10
integer, parameter :: ni=128
integer, parameter :: nj=256
integer, parameter :: nk=64
real(4), dimension(ni,nj,nk) :: a

!$acc enter data copyin(a)

block; use m
!$acc host_data use_device(a)
do nt = 1, ntimes
  call doit(a)
end do
!$acc end host_data
end block

block; use m
do nt = 1, ntimes
  call doit(a)
end do
end block
end

! CHECK: fir.call @_QP__device_sub
! CHECK: fir.call @_QP__host_sub
