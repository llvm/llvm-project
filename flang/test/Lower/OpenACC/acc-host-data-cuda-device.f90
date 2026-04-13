
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

contains

  subroutine vectoraddarray(a, b, n)
    implicit none
    integer :: n
    real, dimension(1:n) :: a, b
  end subroutine 
end module

program testex1
integer, parameter :: ntimes = 10
integer, parameter :: ni=128
integer, parameter :: nj=256
integer, parameter :: nk=64
real(4), dimension(ni,nj,nk) :: a
type :: t
  real(4), dimension(10,10,10) :: a
end type
type(t) :: b
real, dimension(:), allocatable :: a2, b2


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

block; use m
  !$acc host_data use_device(b%a)
  do nt = 1, ntimes
    call doit(b%a)
  end do
  !$acc end host_data
end block

block; use m
  !$acc host_data use_device(b)
  do nt = 1, ntimes
    call doit(b%a)
  end do
  !$acc end host_data
  end block
 
  !$acc host_data use_device(a2, b2) if(allocated(a2))
  call vectoraddarray(a2, b2, 10)
  !$acc end host_data

end

! CHECK: fir.call @_QP__device_sub
! CHECK: fir.call @_QP__host_sub
! CHECK: fir.call @_QP__device_sub
! CHECK: fir.call @_QP__device_sub
