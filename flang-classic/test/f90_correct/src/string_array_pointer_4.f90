!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! Check that the type length of a string array pointer is set correctly.

program string_array_pointer_test_4
  implicit none
  character(:), allocatable, target :: array(:)
  character(:), pointer :: ptr(:)
  array = ['foo']
  ptr(2:2) => array
  ptr(2) = 'bar'
  if (.not. allocated(array)) stop 1
  if (len(ptr) /= 3) stop 2
  if (len(array) /= 3) stop 3
  if (array(1) /= 'bar') stop 4
  deallocate(array)
  print *, 'PASS'
end program
