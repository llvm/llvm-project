! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of TRANSPOSE
module m
  integer, parameter :: matrix(0:1,0:2) = reshape([1,2,3,4,5,6],shape(matrix))
  logical, parameter :: test_transpose_1 = all(transpose(matrix) == reshape([1,3,5,2,4,6],[3,2]))
end module
