!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! If there is no v-list in the edit descriptor or if the data transfer statement
! specifies list-directed or namelist formatting, the processor shall provide v_list
! as a zero-sized array.

MODULE m
  IMPLICIT NONE
  TYPE dt
    INTEGER, ALLOCATABLE :: i
    CONTAINS
      PROCEDURE :: write_dt
      GENERIC :: WRITE(formatted) => write_dt
  END TYPE

  CONTAINS
    SUBROUTINE write_dt(dtv, unit, iotype, v_list, iostat, iomsg)
      CLASS(dt), INTENT(IN) :: dtv
      INTEGER, INTENT(IN) :: unit
      CHARACTER(*), INTENT(IN) :: iotype
      INTEGER, INTENT(IN) :: v_list(:)
      INTEGER, INTENT(OUT) :: iostat
      CHARACTER(*), INTENT(INOUT) :: iomsg
      real, dimension(0) :: a
      if (any(lbound(v_list) .ne. lbound(a))) STOP 1
      if (any(ubound(v_list) .ne. ubound(a))) STOP 2
      if (any(shape(v_list) .ne. shape(a))) STOP 3
      if (size(v_list) .ne. size(a)) STOP 4
      if (any(v_list .ne. a)) STOP 5
    END SUBROUTINE
END MODULE m

program p
  use m

  TYPE(dt) d
  CHARACTER(10) :: internal_file
  INTEGER :: iostat

  WRITE(internal_file, *) d
  print *, 'PASS'
end program
