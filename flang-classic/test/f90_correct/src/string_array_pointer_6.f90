!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! Check that the type length of a string array pointer inside type is set
! correctly when allocating it from a source.

program string_array_pointer_test_6
  implicit none
  character(:), pointer :: array(:, :)
  character(:), pointer :: array2(:)
  type t1
    character(:), pointer :: ptr(:)
  end type t1
  type(t1) :: var1, var2

  allocate(var1%ptr(1), source = "hello world")
  allocate(array2(1), source = "hello world")
  allocate(var2%ptr(1), source = array2)
  allocate(array(1, 3), source = "hello")

  if (len(var1%ptr) /= 11) stop 1
  if (len(var1%ptr(1)) /= 11) stop 2
  if (len(array2) /= 11) stop 3
  if (len(var2%ptr) /= 11) stop 4
  if (len(var2%ptr(1)) /= 11) stop 5
  if (len(array) /= 5) stop 6
  if (all(var1%ptr /= ["hello world"])) stop 7
  if (var1%ptr(1) /= "hello world") stop 8
  if (array2(1) /= "hello world") stop 9
  if (all(var2%ptr /= ["hello world"])) stop 10
  if (var2%ptr(1) /= "hello world") stop 11
  if (array(1, 3) /= "hello") stop 12

  print *, 'PASS'
end program
