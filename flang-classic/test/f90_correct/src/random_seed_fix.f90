!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

!* Test random_seed fix
program test
  integer, parameter :: num = 1
  integer rslts(num), expect(num)
  data expect / 1 /

  call test_with_sized_seed_array()
  call test_with_large_seed_array()
  call test_with_small_seed_array()

  rslts(1) = 1
  call check(rslts, expect, num)
contains
subroutine test_with_sized_seed_array()
  integer :: my_seed_sz
  integer, allocatable :: my_seed_arr(:)
  real :: my_rand

  call random_seed(size=my_seed_sz)

  allocate(my_seed_arr(my_seed_sz))
  my_seed_arr = 0
  my_seed_arr(1) = Z'800000'

  call random_seed(put=my_seed_arr)
  call random_number(my_rand)

  deallocate(my_seed_arr)
end subroutine

subroutine test_with_small_seed_array()
  integer, parameter :: my_seed_sz=8
  integer :: my_seed_arr(my_seed_sz)
  real :: my_rand

  my_seed_arr = 0
  my_seed_arr(1) = Z'800000'

  call random_seed(put=my_seed_arr)
  call random_number(my_rand)
end subroutine

subroutine test_with_large_seed_array()
  integer, parameter :: my_seed_sz=51
  integer :: my_seed_arr(my_seed_sz)
  real :: my_rand

  my_seed_arr = 0
  my_seed_arr(1) = Z'800000'

  call random_seed(put=my_seed_arr)
  call random_number(my_rand)
end subroutine
end program
