! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
!
! Check that repeated do-variable names in nested io-implied-do are diagnosed.
! Section 12.6.3 paragraph 7: The do-variable of an io-implied-do that is in
! another io-implied-do shall not appear as, nor be associated with, the
! do-variable of the containing io-implied-do.

module m
  integer :: mk
end module

program test
  use m, only: mk, mk_alias => mk
  implicit none
  integer :: matrix(10,10), vec(10), cube(5,5,5)
  integer :: i, j, k, ei
  equivalence(j, ei)

  do i = 1, 10
    do j = 1, 10
      matrix(i,j) = 10*(i - 1) + j
    end do
  end do
  vec = [(i, i=1,10)]

  ! OK: different do-variables
  write(*,'(10i5)') ((matrix(i,j), j=1,10), i=1,10)

  ! Bad: j repeated in nested io-implied-do
  !WARNING: I/O implied DO index 'j' appears in an enclosing I/O implied DO loop and should not have the same name [-Wio-implied-do-index-conflict]
  write(*,'(10i5)') ((matrix(i,j), j=1,10), j=1,10)

  ! Bad: i repeated in nested io-implied-do
  !WARNING: I/O implied DO index 'i' appears in an enclosing I/O implied DO loop and should not have the same name [-Wio-implied-do-index-conflict]
  write(*,'(10i5)') ((matrix(i,j), i=1,10), i=1,10)

  ! OK: single (non-nested) io-implied-do
  write(*,'(10i5)') (vec(j), j=1,10)

  ! OK: sibling (non-nested) implied-DOs reusing the same variable
  write(*,'(10i5)') (vec(i), i=1,5), (vec(i), i=6,10)

  ! Bad: j repeated in nested io-implied-do (READ)
  !WARNING: I/O implied DO index 'j' appears in an enclosing I/O implied DO loop and should not have the same name [-Wio-implied-do-index-conflict]
  read(*,*) ((matrix(i,j), j=1,10), j=1,10)

  ! Bad: ei is associated with j via EQUIVALENCE
  !WARNING: I/O implied DO index 'ei' should not be associated with do-variable 'j' of an enclosing I/O implied DO loop [-Wio-implied-do-index-conflict]
  write(*,'(10i5)') ((matrix(i,ei), ei=1,10), j=1,10)

  ! Bad: USE-renamed mk_alias is the same variable as mk
  !WARNING: I/O implied DO index 'mk_alias' should not be associated with do-variable 'mk' of an enclosing I/O implied DO loop [-Wio-implied-do-index-conflict]
  write(*,'(10i5)') ((matrix(i,mk_alias), mk_alias=1,10), mk=1,10)

  ! Bad: triple nesting, k repeated at innermost level
  !WARNING: I/O implied DO index 'k' appears in an enclosing I/O implied DO loop and should not have the same name [-Wio-implied-do-index-conflict]
  write(*,*) (((cube(i,j,k), k=1,5), j=1,5), k=1,5)

  ! Bad: ASSOCIATE — x is associated with j
  associate(x => j)
    !WARNING: I/O implied DO index 'x' should not be associated with do-variable 'j' of an enclosing I/O implied DO loop [-Wio-implied-do-index-conflict]
    write(*,'(10i5)') ((matrix(i,x), x=1,10), j=1,10)
  end associate
end program test
