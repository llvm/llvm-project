! RUN: %python %S/test_errors.py %s %flang_fc1
! F'2023 19.4 p5: a data-implied-do index variable without an explicit
! integer-type-spec takes the type that its name would have as a variable
! of the scoping unit, and the type declaration statement establishing that
! type may appear later in the same specification part than the DATA
! statement.

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
