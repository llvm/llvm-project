!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

subroutine rank2(a, b, c)
  real*4, target, contiguous :: a(:,:)
  real*4, target, contiguous :: b(:,:)
  real*4, target, contiguous :: c(:,:)

  c = matmul(a,b)
end subroutine

subroutine to_1_dim(src, dest, n) 
   integer :: n
   real*4 :: src(*)
   real*4 :: dest(*)
   dest(1:n) = src(1:n)
end subroutine

program p
 implicit none
 interface
  subroutine rank2(a, b, c)
    real*4, contiguous :: a(:,:)
    real*4, contiguous :: b(:,:)
    real*4, contiguous :: c(:,:)
  end subroutine
 end interface

  integer, parameter :: N=7*48
  integer :: tstnbr
  integer :: rslt_sz

  real*4 :: result(N)
  real*4 :: expect(N)

  integer, parameter :: o_extent=2
  integer, parameter :: n_extent=6
  integer, parameter :: m_extent=4
  integer, parameter :: k_extent=8

  real*4, target,dimension(n_extent,m_extent) :: arr1
  real*4, target, dimension(m_extent,k_extent) :: arr2
  real*4, target, dimension(n_extent,k_extent) :: arr3

  real*4, contiguous, pointer, dimension(:,:) :: arr1_ptr
  real*4, contiguous, pointer, dimension(:,:) :: arr2_ptr
  real*4, contiguous, pointer, dimension(:,:) :: arr3_ptr

  data arr1 /0,1,2,3,4,5,6,7,8,9,10,11,                 &
             12,13,14,15,16,17,18,19,20,22,22,23/
  data arr2 /0,1,2,3,4,5,6,7,8,9,10,11,                 &
             12,13,14,15,16,17,18,19,20,21,22,23,               &
             24,25,26,27,28,29,30,31/
  data arr3 /0,1,2,3,4,5,6,7,8,9,10,11,                         &
             12,13,14,15,16,17,18,19,20,21,22,23,               &
             24,25,26,27,28,29,30,31,32,33,34,35,               &
             36,37,38,39,40,41,42,43,44,45,46,47/

  data expect  /  &
  ! test 1
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 372.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 786.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 2
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 372.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 786.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 3
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 372.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 786.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 4
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 372.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 786.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 5
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 372.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 786.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 6
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 372.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 786.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0, &
  ! test 7
      84.0, 90.0, 96.0, 105.0, 108.0, 114.0, 228.0, 250.0, 272.0, &
      301.0, 316.0, 338.0, 372.0, 410.0, 448.0, 497.0, 524.0, 562.0, &
      516.0, 570.0, 624.0, 693.0, 732.0, 786.0, 660.0, 730.0, 800.0, &
      889.0, 940.0, 1010.0, 804.0, 890.0, 976.0, 1085.0, 1148.0, 1234.0, &
      948.0, 1050.0, 1152.0, 1281.0, 1356.0, 1458.0, 1092.0, 1210.0, 1328.0, &
      1477.0, 1564.0, 1682.0 /



  tstnbr = 1
  call rank2(arr1, arr2, arr3)
  rslt_sz = size(arr3)
  call to_1_dim(arr3, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz+1), rslt_sz)

  tstnbr = 2
  arr1_ptr => arr1
  arr2_ptr => arr2
  arr3_ptr => arr3
  arr3_ptr = matmul(arr1_ptr,arr2_ptr)
  rslt_sz = size(arr3_ptr)
  call to_1_dim(arr3_ptr, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz+1), rslt_sz)
!  print *,"arr3_ptr"
!  print *,arr3_ptr

  tstnbr = 3
  arr3_ptr = 0
  call rank2(arr1_ptr,arr2_ptr,arr3_ptr)
  rslt_sz = size(arr3_ptr)
  call to_1_dim(arr3_ptr, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz+1), rslt_sz)
!  print *,"arr3_ptr"
!  print *,arr3_ptr

  tstnbr = 4
  arr3_ptr = 0
  call rank2(arr1_ptr,arr2,arr3_ptr)
  rslt_sz = size(arr3_ptr)
  call to_1_dim(arr3_ptr, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz+1), rslt_sz)
!  print *,"arr3_ptr"
!  print *,arr3_ptr

  tstnbr = 5
  arr3_ptr = 0
  call rank2(arr1_ptr,arr2,arr3_ptr)
  rslt_sz = size(arr3_ptr)
  call to_1_dim(arr3_ptr, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz+1), rslt_sz)
!  print *,"arr3_ptr"
!  print *,arr3_ptr

  tstnbr = 6
  arr3_ptr = 0
  call rank2(arr1,arr2_ptr,arr3_ptr)
  rslt_sz = size(arr3)
  call to_1_dim(arr3, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz+1), rslt_sz)
!  print *,"arr3_ptr"
!  print *,arr3_ptr

  tstnbr = 7
  allocate(arr1_ptr(n_extent,m_extent), arr2_ptr(m_extent,k_extent), arr3_ptr(n_extent,k_extent) )
  arr1_ptr = arr1
  arr2_ptr = arr2
  arr3_ptr = arr3
  arr3_ptr = matmul(arr1_ptr,arr2_ptr)
  rslt_sz = size(arr3_ptr)
  call to_1_dim(arr3_ptr, result((tstnbr-1)*rslt_sz+1:tstnbr*rslt_sz+1), rslt_sz)
!  print *,"arr3_ptr"
!  print *,arr3_ptr

  call check(result, expect, N)
!  print *, result
end program
