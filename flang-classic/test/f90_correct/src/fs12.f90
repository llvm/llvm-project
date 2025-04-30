!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

!pgi$g opt 2
subroutine contig_cpy(src, dest)
  real*4, target, contiguous :: src(:,:)
  real*4, target, contiguous :: dest(:,:)

  dest = src
end subroutine

subroutine to_1_dim(src, dest, n)
   integer :: n
   real*4 :: src(*)
   real*4 :: dest(*)
!   print *, n
!   print *, src(1:n)
   dest(1:n) = src(1:n)
!   print *, dest(1:n)
end subroutine

program p
 implicit none
 interface
  subroutine contig_cpy(a, b)
    real*4, contiguous :: a(:,:)
    real*4, contiguous :: b(:,:)
  end subroutine
 end interface

  integer, parameter :: N=4*24
  integer :: tstnbr
  integer :: rslt_sz

  real*4 :: result(N)
  real*4 :: expect(N)


  integer, parameter :: n_extent=6
  integer, parameter :: m_extent=4

  real*4, target, dimension(n_extent,m_extent) :: arr1
  real*4, target, dimension(n_extent,m_extent) :: arr2

  real*4, contiguous, pointer, dimension(:,:) :: arr1_ptr

  data arr1 /0,1,2,3,4,5,6,7,8,9,10,11,                 &
             12,13,14,15,16,17,18,19,20,22,22,23/

  data expect  /  &
       0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,22,23, &
       0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,22,23, &
       0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,22,23, &
       0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,22,23  /

  tstnbr = 1
  arr2 = 0
  call contig_cpy(arr1, arr2)
  rslt_sz = size(arr2)
  call to_1_dim(arr2, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz), rslt_sz)

  tstnbr = 2
  arr2 = 0
  arr1_ptr=>arr2
  call contig_cpy(arr1, arr1_ptr)
  rslt_sz = size(arr1_ptr)
  call to_1_dim(arr1_ptr, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz), rslt_sz)

  tstnbr = 3
  arr2 = 0
  arr1_ptr=>arr1
  arr2 = arr1_ptr
  rslt_sz = size(arr2)
  call to_1_dim(arr2, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz), rslt_sz)

  tstnbr = 4
  allocate(arr1_ptr(n_extent,m_extent))
  arr1_ptr = 0
  arr1_ptr = arr1
  rslt_sz = size(arr1_ptr)
  call to_1_dim(arr1_ptr, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz), rslt_sz)

  call check(result, expect, N)
!  print *, result

end program
