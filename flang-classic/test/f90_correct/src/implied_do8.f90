!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for loop body of implied-do loop containing function call that some of
! actual arguments are array constructor.

module m
  implicit none
contains
  subroutine test1()
    integer :: i, j, k, ii
    logical :: rslt(63)
    logical :: expect(63)

    rslt = [(((foo([i > j]), i = 1, 3), (foo([i > k]), i = 1, 3), &
              foo([j > k]), j = 1, 3), k = 1, 3)]
    ii = 1
    do k = 1, 3
      do j = 1, 3
        do i = 1, 3
          expect(ii) = foo([i > j])
          ii = ii + 1
        enddo
        do i = 1, 3
          expect(ii) = foo([i > k])
          ii = ii + 1
        enddo
        expect(ii) = foo([j > k])
        ii = ii + 1
      enddo
    enddo
    if (any(rslt .neqv. expect)) STOP 1
  end subroutine

  subroutine test2()
    integer :: i, j, k, ii
    logical, allocatable :: rslt(:)
    logical :: expect(26)
    rslt = [(((foo([i > j]), i = 1, j), (foo([i > k]), i = 1, j), &
              foo([j > k]), j = 1, k), k = 1, 3)]
    if (sizeof(rslt) /= sizeof(expect)) STOP 2
    ii = 1
    do k = 1, 3
      do j = 1, k
        do i = 1, j
          expect(ii) = foo([i > j])
          ii = ii + 1
        enddo
        do i = 1, j
          expect(ii) = foo([i > k])
          ii = ii + 1
        enddo
        expect(ii) = foo([j > k])
        ii = ii + 1
      enddo
    enddo
    if (any(rslt .neqv. expect)) STOP 3
  end subroutine

  function foo(x) result(z)
    logical :: x(:)
    logical :: z
    z = x(1)
  end function
end module

program p
  use m
  implicit none
  call test1()
  call test2()
  print *, "PASS"
end program
