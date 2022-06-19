! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of UNPACK (valid cases)
module m
  integer, parameter :: vector(*) = [1, 2, 3, 4]
  integer, parameter :: field(2,3) = reshape([(-j,j=1,6)], shape(field))
  logical, parameter :: mask(*,*) = reshape([.false., .true., .true., .false., .false., .true.], shape(field))
  logical, parameter :: test_unpack_1 = all(unpack(vector, mask, 0) == reshape([0,1,2,0,0,3], shape(mask)))
  logical, parameter :: test_unpack_2 = all(unpack(vector, mask, field) == reshape([-1,1,2,-4,-5,3], shape(mask)))
end module
