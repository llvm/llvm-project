!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! Check that the type length of a string array pointer inside type is set
! correctly when pointing to a target.

program string_array_pointer_test_7
  implicit none
  character(:), allocatable, target :: array(:)
  type t1
    character(:), pointer :: ptr(:)
  end type t1
  type(t1) :: var

  allocate(array(1), source = "hello world")
  var%ptr(1:1)=>array

  if (len(array) /= 11) stop 1
  if (len(var%ptr) /= 11) stop 2
  if (len(var%ptr(1)) /= 11) stop 3
  if (array(1) /= "hello world") stop 4
  if (all(var%ptr /= ["hello world"])) stop 5
  if (var%ptr(1) /= "hello world") stop 6

  print *, 'PASS'
end program
