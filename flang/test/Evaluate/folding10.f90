! RUN: %S/test_folding.sh %s %t %flang_fc1
! REQUIRES: shell
! Tests folding of SHAPE(TRANSFER(...))

module m
  logical, parameter :: test_size_1 = size(shape(transfer(123456789,0_1,size=4))) == 1
  logical, parameter :: test_size_2 = all(shape(transfer(123456789,0_1,size=4)) == [4])
  logical, parameter :: test_scalar_1 = size(shape(transfer(123456789, 0_1))) == 0
  logical, parameter :: test_vector_1 = size(shape(transfer(123456789, [0_1]))) == 1
  logical, parameter :: test_vector_2 = all(shape(transfer(123456789, [0_1])) == [4])
  logical, parameter :: test_array_1 = size(shape(transfer(123456789, reshape([0_1],[1,1])))) == 1
  logical, parameter :: test_array_2 = all(shape(transfer(123456789, reshape([0_1],[1,1]))) == [4])
  logical, parameter :: test_array_3 = all(shape(transfer([1.,2.,3.], [(0.,0.)])) == [2])
end module
