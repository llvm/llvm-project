! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of SHAPE(TRANSFER(...))
! Adjusted to allow for folding (or not) of TRANSFER().

module m
  integer :: j
  real :: a(3)
  logical, parameter :: test_size_v1 = size(shape(transfer(j, 0_1,size=4))) == 1
  logical, parameter :: test_size_v2 = all(shape(transfer(j, 0_1,size=4)) == [4])
  logical, parameter :: test_scalar_v1 = size(shape(transfer(j, 0_1))) == 0
  logical, parameter :: test_vector_v1 = size(shape(transfer(j, [0_1]))) == 1
  logical, parameter :: test_vector_v2 = all(shape(transfer(j, [0_1])) == [4])
  logical, parameter :: test_array_v1 = size(shape(transfer(j, reshape([0_1],[1,1])))) == 1
  logical, parameter :: test_array_v2 = all(shape(transfer(j, reshape([0_1],[1,1]))) == [4])
  logical, parameter :: test_array_v3 = all(shape(transfer(a, [(0.,0.)])) == [2])

  logical, parameter :: test_size_1 = size(shape(transfer(123456789,0_1,size=4))) == 1
  logical, parameter :: test_size_2 = all(shape(transfer(123456789,0_1,size=4)) == [4])
  logical, parameter :: test_scalar_1 = size(shape(transfer(123456789, 0_1))) == 0
  logical, parameter :: test_vector_1 = size(shape(transfer(123456789, [0_1]))) == 1
  logical, parameter :: test_vector_2 = all(shape(transfer(123456789, [0_1])) == [4])
  logical, parameter :: test_array_1 = size(shape(transfer(123456789, reshape([0_1],[1,1])))) == 1
  logical, parameter :: test_array_2 = all(shape(transfer(123456789, reshape([0_1],[1,1]))) == [4])
  logical, parameter :: test_array_3 = all(shape(transfer([1.,2.,3.], [(0.,0.)])) == [2])
end module
