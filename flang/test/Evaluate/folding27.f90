! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of CSHIFT (valid cases)
module m
  integer, parameter :: arr(2,3) = reshape([1, 2, 3, 4, 5, 6], shape(arr))
  logical, parameter :: test_sanity = all([arr] == [1, 2, 3, 4, 5, 6])
  logical, parameter :: test_cshift_0 = all(cshift([1, 2, 3], 0) == [1, 2, 3])
  logical, parameter :: test_cshift_1 = all(cshift([1, 2, 3], 1) == [2, 3, 1])
  logical, parameter :: test_cshift_2 = all(cshift([1, 2, 3], 3) == [1, 2, 3])
  logical, parameter :: test_cshift_3 = all(cshift([1, 2, 3], 4) == [2, 3, 1])
  logical, parameter :: test_cshift_4 = all(cshift([1, 2, 3], -1) == [3, 1, 2])
  logical, parameter :: test_cshift_5 = all([cshift(arr, 1, dim=1)] == [2, 1, 4, 3, 6, 5])
  logical, parameter :: test_cshift_6 = all([cshift(arr, 1, dim=2)] == [3, 5, 1, 4, 6, 2])
  logical, parameter :: test_cshift_7 = all([cshift(arr, [1, 2, 3])] == [2, 1, 3, 4, 6, 5])
  logical, parameter :: test_cshift_8 = all([cshift(arr, [1, 2], dim=2)] == [3, 5, 1, 6, 2, 4])
end module
