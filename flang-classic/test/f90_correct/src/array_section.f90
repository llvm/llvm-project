! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program p
  type t
    integer :: n
  end type
  integer :: i, j
  type(t) :: a(10)
  integer :: result(21), expect(21)
  data expect /1, 2, 3, 4, 5, 6, 7, 8, 9, 10, &
               3, 4, 5, 6, 7, 8, 9, &
               5, 6, 7, 8 /
  result = 0
  j = 1
  forall (i = 1 : 10)
    a(i)%n = i
  end forall

  call sub1(a)

  if(any(result.ne.expect)) STOP 1
  print *,'PASS'

contains
  recursive  subroutine sub1(arr)
    class(t), dimension(:) :: arr
    integer :: k
    if (size(arr) < 2) return

    do k = 1, size(arr)
      result(j) = arr(k)%n
      j = j + 1
    end do

    call sub1(arr(3:size(arr)-1))
  end subroutine
end program
