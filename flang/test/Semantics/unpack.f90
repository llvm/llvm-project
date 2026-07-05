! RUN: %python %S/test_errors.py %s %flang_fc1
! UNPACK() intrinsic function error tests
program test_unpack
  integer, dimension(2) :: vector = [343, 512]
  logical, dimension(2, 2) :: mask = &
    reshape([.true., .false., .true., .false.], [2, 2])
  integer, dimension(2, 2) :: field = reshape([1, 2, 3, 4, 5, 6], [2, 2])
  integer, dimension(2, 1) :: bad_field = reshape([1, 2], [2, 1])
  integer :: scalar_field
  integer, dimension(2, 2) :: result
  result = unpack(vector, mask, field)
  !ERROR: Dimension 2 of 'mask=' argument has extent 2, but 'field=' argument has extent 1
  result = unpack(vector, mask, bad_field)
  result = unpack(vector, mask, scalar_field)
end program
