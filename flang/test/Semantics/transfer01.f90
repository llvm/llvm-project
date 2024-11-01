! RUN: %python %S/test_errors.py %s %flang_fc1
! Check errors in TRANSFER()

subroutine subr(o)
  integer, intent(in), optional :: o
  type empty
  end type
  type(empty) :: empty1(1)
  real :: empty2(0)
  character(0) :: empty3(1)
  integer, pointer :: source(:)
  integer, allocatable :: ia
  integer, pointer :: ip
  !ERROR: Element size of MOLD= array may not be zero when SOURCE= is not empty
  print *, transfer(1., empty1)
  print *, transfer(1., empty2) ! ok
  !ERROR: Element size of MOLD= array may not be zero when SOURCE= is not empty
  print *, transfer(1., empty3)
  !WARNING: Element size of MOLD= array may not be zero unless SOURCE= is empty
  print *, transfer(source, empty1)
  print *, transfer(source, empty2) ! ok
  !WARNING: Element size of MOLD= array may not be zero unless SOURCE= is empty
  print *, transfer(source, empty3)
  !ERROR: SIZE= argument may not be the optional dummy argument 'o'
  print *, transfer(1., empty2, size=o)
  !WARNING: SIZE= argument that is allocatable or pointer must be present at execution; parenthesize to silence this warning
  print *, transfer(1., empty2, size=ia)
  !WARNING: SIZE= argument that is allocatable or pointer must be present at execution; parenthesize to silence this warning
  print *, transfer(1., empty2, size=ip)
end

