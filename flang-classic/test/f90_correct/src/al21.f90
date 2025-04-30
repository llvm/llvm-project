! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Test assignment of array-valued intrinsics with F2003 allocatable
! assignment semantics. Compile with -Mallocatable=03.
program al21
  implicit none
  logical :: fail = .false.
  integer :: i, j
  real :: a1(16) = [ (i, i=1, 16) ]
  real :: a2(4,4)
  a2 = reshape(a1, [4, 4])

  call test_matmul()
  call test_minval()
  call test_product()
  call test_sum()
  call test_cshift()
  call test_eoshift()
  call test_count()
  if (.not. fail) write(*,'("PASS")')

contains

  subroutine test_matmul()
    real, allocatable :: b2(:,:)
    b2 = matmul(a2, a2)
    call check('matmul', reshape(b2, [16]), [ &
       90.0, 100.0, 110.0, 120.0, &
      202.0, 228.0, 254.0, 280.0, &
      314.0, 356.0, 398.0, 440.0, &
      426.0, 484.0, 542.0, 600.0 ])
  end subroutine

  subroutine test_minval()
    real, allocatable :: b1(:)
    b1 = minval(a2, 1)
    call check('minval', b1, [1.0, 5.0, 9.0, 13.0])
  end subroutine

  subroutine test_cshift()
    real, allocatable :: b1(:)
    b1 = cshift(a1, 4)
    call check('cshift', b1, [ &
       5.0,  6.0,  7.0,  8.0, 9.0, 10.0, 11.0, 12.0, &
      13.0, 14.0, 15.0, 16.0, 1.0,  2.0,  3.0,  4.0  &
    ])
  end subroutine

  subroutine test_product()
    real, allocatable :: b1(:)
    b1 = product(a2, 1)
    call check('product', b1, [24.0, 1680.0, 11880.0, 43680.0])
  end subroutine

  subroutine test_sum()
    real, allocatable :: b1(:)
    b1 = sum(a2, 1)
    call check('sum', b1, [10.0, 26.0, 42.0, 58.0])
  end subroutine

  subroutine test_eoshift()
    real, allocatable :: b1(:)
    b1 = eoshift(a1, 4)
    call check('eoshift', b1, [ &
       5.0,  6.0,  7.0,  8.0, 9.0, 10.0, 11.0, 12.0, &
      13.0, 14.0, 15.0, 16.0, 0.0,  0.0,  0.0,  0.0  &
    ])
  end subroutine

  subroutine test_count()
    logical :: mask(4, 4)
    integer, allocatable :: n(:)
    do i = 1, 4
      do j = 1, 4
        mask(i, j) = mod(i, 2) == 0 .and. mod(j, 2) == 0
      end do
    end do
    n = count(mask, 1)
    call checki('count', n, [0, 2, 0, 2])
  end subroutine

  ! Check that actual is the same as expected; report failure if not.
  subroutine check(label, actual, expected)
    character(len=*) :: label
    real :: actual(:)
    real :: expected(:)
    if (size(actual) /= size(expected)) then
    else if (any(actual /= expected)) then
    else
      return
    end if
    write(*,'("FAIL: ",a)') label
    print *," expected:", expected
    print *," actual:  ", actual
    fail = .true.
  end subroutine

  ! Check that actual is the same as expected; report failure if not.
  subroutine checki(label, actual, expected)
    character(len=*) :: label
    integer :: actual(:)
    integer :: expected(:)
    if (size(actual) /= size(expected)) then
    else if (any(actual /= expected)) then
    else
      return
    end if
    write(*,'("FAIL: ",a)') label
    print *," expected:", expected
    print *," actual:  ", actual
    fail = .true.
  end subroutine

end
