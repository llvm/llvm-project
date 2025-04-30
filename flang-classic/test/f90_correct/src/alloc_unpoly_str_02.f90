!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! related to Flang github issue #713

program test
  implicit none
  class(*), allocatable :: a(:, :)
  character(:), allocatable :: b(:, :)
  character(:), allocatable :: c(:, :)

  b = reshape(['ab', 'cd', 'ef', 'gh', 'ij', 'kl'], [2, 3])
  c = reshape(['abc', 'def', 'ghi', 'jkl', 'imn', 'opq'], [3, 2])
  if (len(b) /= 2) stop 1

  allocate(a, source = b)

  select type (a)
  type is (character(*))
    if (len(a) /= len(b)) stop 2
    if (size(a) /= size(b)) stop 3
    if (any(shape(a) /= shape(b))) stop 4
    if (any(a /= b)) stop 5
  class default
    stop 6
  end select
  deallocate(a)

  call sub(a, c)
  print *, 'PASS'

contains
  subroutine sub(x, y)
    class(*), allocatable :: x(:, :)
    character(:), allocatable :: y(:, :)

    allocate(character(len(y)) :: x(lbound(y, 1) : ubound(y, 1),&
                                    lbound(y, 2) : ubound(y, 2)))
    select type (x)
    type is (character(*))
      if (len(x) /= len(y)) stop 7
      if (size(x) /= size(y)) stop 8
      if (any(shape(x) /= shape(y))) stop 9
    class default
      stop 10
    end select
    deallocate(x)
  end subroutine
end
