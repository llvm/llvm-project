!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for implied-do loop containing function call that some of actual
! arguments are array constructor.

module m
  implicit none
contains
  subroutine test1()
    integer :: i, j, k, ii
    integer, allocatable :: rslt(:)
    integer :: expect(45)

    rslt = [(((9, i = 1, foo([j])), (99, i = 1, foo([k])), 999, j = 1, 3), &
             k = 1, 3)]
    if (sizeof(rslt) /= sizeof(expect)) STOP 1
    ii = 1
    do k = 1, 3
      do j = 1, 3
        do i = 1, foo([j])
          expect(ii) = 9
          ii = ii + 1
        enddo
        do i = 1, foo([k])
          expect(ii) = 99
          ii = ii + 1
        enddo
        expect(ii) = 999
        ii = ii + 1
      enddo
    enddo
    if (any(rslt /= expect)) STOP 2
  end subroutine

  subroutine test2()
    integer :: i, j, k, ii
    integer, allocatable :: rslt(:)
    integer :: expect(45)

    rslt = [(((9 * foo([i]), i = 1, foo([j])), &
              (99 * foo([k]), i = 1, foo([k])), 999 * foo([k]), j = 1, 3), &
            k = 1, 3)]
    if (sizeof(rslt) /= sizeof(expect)) STOP 3
    ii = 1
    do k = 1, 3
      do j = 1, 3
        do i = 1, foo([j])
          expect(ii) = 9 * foo([i])
          ii = ii + 1
        enddo
        do i = 1, foo([k])
          expect(ii) = 99 * foo([k])
          ii = ii + 1
        enddo
        expect(ii) = 999 * foo([k])
        ii = ii + 1
      enddo
    enddo
    if (any(rslt /= expect)) STOP 4
  end subroutine

  subroutine test3()
    integer :: i, j, k, ii
    integer, allocatable :: rslt(:)
    integer :: expect(40)

    rslt = [((9 * foo([j]), 1 : foo([k]), (99 * foo([k]), i = 1, foo([k])), &
              999 * foo([k]), j = 1, foo([k])), k = 1, 3)]
    if (sizeof(rslt) /= sizeof(expect)) STOP 5
    ii = 1
    do k = 1, 3
      do j = 1, foo([k])
        expect(ii) = 9 * foo([j])
        ii = ii + 1
        do i = 1, foo([k])
          expect(ii) = i
          ii = ii + 1
        enddo
        do i = 1, foo([k])
          expect(ii) = 99 * foo([k])
          ii = ii + 1
        enddo
        expect(ii) = 999 * foo([k])
        ii = ii + 1
      enddo
    enddo
    if (any(rslt /= expect)) STOP 6
  end subroutine

  subroutine test4()
    integer :: i, j, k, ii, jj
    integer, allocatable :: rslt(:)
    integer :: expect(63)

    rslt = [(((9, i = 1, foo([j])), ((ii, ii = 1, i), i = 1, foo([k])), 999, &
              j = 1, 3), (99, i = 1, foo([k])), k = 1, 3)]

    if (sizeof(rslt) /= sizeof(expect)) STOP 7
    jj = 1
    do k = 1, 3
      do j = 1, 3
        do i = 1, foo([j])
          expect(jj) = 9
          jj = jj + 1
        enddo
        do i = 1, foo([k])
          do ii = 1, i
            expect(jj) = ii
            jj = jj + 1
          enddo
        enddo
        expect(jj) = 999
        jj = jj + 1
      enddo
      do i = 1, foo([k])
        expect(jj) = 99
        jj = jj + 1
      enddo
    enddo
    if (any(rslt /= expect)) STOP 8
  end subroutine

  function foo(x) result(z)
    integer :: x(:)
    integer :: z
    z = x(1)
  end function
end module

program p
  use m
  implicit none
  call test1()
  call test2()
  call test3()
  call test4()
  print *, "PASS"
end program
