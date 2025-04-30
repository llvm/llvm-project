!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! related to Flang github issue #713

program test
  implicit none
  class(*), allocatable :: a
  character(:), allocatable :: b
  character(:), allocatable :: c

  b = 'ab'
  c = 'abc'
  if (len(b) /= 2) stop 1

  allocate(a, source = b)

  select type (a)
  type is (character(*))
    if (len(a) /= len(b)) stop 2
    if (a /= b) stop 3
  class default
    stop 4
  end select
  deallocate(a)

  call sub(a, c)
  print *, 'PASS'

contains
  subroutine sub(x, y)
    class(*), allocatable :: x
    character(:), allocatable :: y

    allocate(character(len(y)) :: x)
    select type (x)
    type is (character(*))
      if (len(x) /= len(y)) stop 5
    class default
      stop 6
    end select
    deallocate(x)
  end subroutine
end

