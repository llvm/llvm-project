!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Checking for polymorphic pointer inside type pointing to string pointer.

program poly_ptr_to_str_2

  type t
    class(*), pointer :: val
  end type t

  type(t) :: x
  character(3), target :: a = 'foo'
  character(:), pointer :: b
  character(:), allocatable :: s

  b => a
  x%val => b

  select type (val => x%val)
  type is (character(*))
    s = val
  end select

  if (.not.allocated(s)) stop 1
  if (len(s) /= 3) stop 2
  if (s /= 'foo') stop 3

  write(*, *) 'PASS'
end

