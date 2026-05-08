
! RUN: bbc -fopenacc -fcuda -emit-hlfir %s -o - | FileCheck %s

module m

real, allocatable, pinned :: pinned_real(:,:,:)

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
type t
  integer, pointer :: p1
  integer, pointer :: p2
end type
type t2
  type(t) :: obj
end type
interface foo
subroutine foo_device(p)
  integer, pointer, device :: p
end subroutine
subroutine foo_host(p)
  integer, pointer :: p
end subroutine
end interface

interface foo_array
  subroutine foo_device_array(x)
    real, device :: x(:,:)
  end
  subroutine foo_host_array(x)
    real :: x(:,:)
  end
end interface

contains

  subroutine test(obj)
    type(t) :: obj
    type(t2) :: obj2
    !$acc host_data use_device(obj%p1)
    call foo(obj%p1)
    call foo(obj%p2)
    !$acc end host_data

    call foo(obj%p1)

    !$acc host_data use_device(obj%p1, obj%p2)
    call foo(obj%p1)
    call foo(obj%p2)
    !$acc end host_data

    !$acc host_data use_device(obj2%obj%p1)
    call foo(obj2%obj%p1)
    call foo(obj2%obj%p2)
    !$acc end host_data
  end subroutine
! CHECK-LABEL: func.func @_QMmPtest
! CHECK: fir.call @_QPfoo_device
! CHECK: fir.call @_QPfoo_host
! CHECK: fir.call @_QPfoo_host
! CHECK: fir.call @_QPfoo_device
! CHECK: fir.call @_QPfoo_device
! CHECK: fir.call @_QPfoo_device
! CHECK: fir.call @_QPfoo_host

  subroutine test_array(a, i)
    real :: a(4,4,4)
    integer :: i
    !$acc host_data use_device(a(:,:,i))
    call foo_array(a(:,:,i))
    !$acc end host_data
  end subroutine

! CHECK-LABEL: func.func @_QMmPtest_array
! CHECK: fir.call @_QPfoo_device_array

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

subroutine test_use_details()
  use m
  call doit(pinned_real)
  !$acc host_data use_device(pinned_real)
  call doit(pinned_real)
  !$acc end host_data
  call doit(pinned_real)
end subroutine

! CHECK: fir.address_of(@_QP__host_sub)
! CHECK: fir.address_of(@_QP__device_sub)
! CHECK: fir.address_of(@_QP__host_sub)
