! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! The bounds of x are changed in an internal subroutine so
! they can't be propagated into shapes.  y is not a problem.
program al20

  integer, allocatable :: x(:), y(:)

  allocate(x(1:2))
  allocate(y(1:2))
  x = 1
  y = 2
  call check(x, [1, 1], 2)
  call check(y, [2, 2], 2)
  call sub()
  x = 3
  y = 4
  call check(x, [3, 3, 3], 3)
  call check(y, [4, 4], 2)

contains

  subroutine sub
    deallocate(x)
    allocate(x(3:5))
  end subroutine

end program
