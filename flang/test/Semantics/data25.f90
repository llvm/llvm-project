! RUN: %python %S/test_errors.py %s %flang_fc1
! F'2023 19.4 p5: a data-implied-do index without an integer-type-spec
! takes the type its name has in the scoping unit; the declaration may
! follow the DATA statement in the same specification part.

! Index declared after the DATA statement: conforming, no error.
subroutine s1
  implicit none
  logical, dimension(4), save :: util
  data (util(i),i=1,4)/4*.true./
  integer :: i
end subroutine

! Nested implied-DOs, both indices declared later.
subroutine s2
  implicit none
  logical, dimension(2,2), save :: m
  data ((m(i,j),i=1,2),j=1,2)/4*.true./
  integer :: i, j
end subroutine

! Index never declared under IMPLICIT NONE(TYPE): still an error.
subroutine s3
  implicit none
  logical, dimension(4), save :: util
  !ERROR: No explicit type declared for 'i'
  data (util(i),i=1,4)/4*.true./
end subroutine

! The index's type, wherever its name is declared, must be integer.
subroutine s4
  implicit none
  logical, dimension(4), save :: util
  !ERROR: Must have INTEGER type, but is CHARACTER(KIND=1,LEN=1_8)
  data (util(i),i=1,4)/4*.true./
  character :: i
end subroutine

! Execution-part DATA statement (obsolescent placement): the whole
! specification part precedes it, so an undeclared index is still an error.
subroutine s5
  implicit none
  logical, dimension(4), save :: util
  continue
  !ERROR: No explicit type declared for 'i'
  data (util(i),i=1,4)/4*.true./
end subroutine

! DATA in a BLOCK construct's specification part: the index may be
! declared later in the same block specification part.
subroutine s6
  implicit none
  block
    logical, dimension(4), save :: util
    data (util(i),i=1,4)/4*.true./
    integer :: i
  end block
end subroutine

! The deferral is confined to the DATA statement's object list: a
! standalone ac-implied-do in a later declaration's initializer does not
! acquire it.
subroutine s7
  implicit none
  integer :: i
  logical, dimension(4), save :: util
  data (util(i),i=1,4)/4*.true./
  !ERROR: No explicit type declared for 'j'
  integer :: a(4) = [(j, j=1,4)]
end subroutine

! DATA in a BLOCK DATA subprogram.
block data s8
  implicit none
  logical, dimension(4) :: util
  common /cb/ util
  data (util(i),i=1,4)/4*.true./
  integer :: i
end block data
