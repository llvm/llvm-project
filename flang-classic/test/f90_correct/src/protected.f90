!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for protected attribute

module m
  implicit none
  integer, protected, pointer :: v
  integer, protected :: v1
end module m

program p
  use m
  implicit none
  integer, target :: a
  !{error "PGF90-S-0155-A use-associated object with the PROTECTED attribute cannot be an actual argument when the dummy argument is INTENT(OUT) or INTENT(INOUT) - v1"}
  call reset1(v1)
  !{error "PGF90-S-0155-A use-associated object with the PROTECTED attribute cannot be an actual argument when the dummy argument is INTENT(OUT) or INTENT(INOUT) - v1"}
  call reset2(v1)
  !{error "PGF90-S-0155-A use-associated object with the PROTECTED attribute cannot be assigned - v"}
  v => a
contains
  subroutine reset1(z)
    integer, intent(inout) :: z
  end subroutine
  subroutine reset2(z)
    integer, intent(out) :: z
  end subroutine
end program p
