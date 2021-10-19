! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of EOSHIFT (valid cases)
module m
  integer, parameter :: arr(2,3) = reshape([1, 2, 3, 4, 5, 6], shape(arr))
  logical, parameter :: test_sanity = all([arr] == [1, 2, 3, 4, 5, 6])
  logical, parameter :: test_eoshift_0 = all(eoshift([1, 2, 3], 0) == [1, 2, 3])
  logical, parameter :: test_eoshift_1 = all(eoshift([1, 2, 3], 1) == [2, 3, 0])
  logical, parameter :: test_eoshift_2 = all(eoshift([1, 2, 3], -1) == [0, 1, 2])
  logical, parameter :: test_eoshift_3 = all(eoshift([1., 2., 3.], 1) == [2., 3., 0.])
  logical, parameter :: test_eoshift_4 = all(eoshift(['ab', 'cd', 'ef'], -1, 'x') == ['x ', 'ab', 'cd'])
  logical, parameter :: test_eoshift_5 = all([eoshift(arr, 1, dim=1)] == [2, 0, 4, 0, 6, 0])
  logical, parameter :: test_eoshift_6 = all([eoshift(arr, 1, dim=2)] == [3, 5, 0, 4, 6, 0])
  logical, parameter :: test_eoshift_7 = all([eoshift(arr, [1, -1, 0])] == [2, 0, 0, 3, 5, 6])
  logical, parameter :: test_eoshift_8 = all([eoshift(arr, [1, -1], dim=2)] == [3, 5, 0, 0, 2, 4])
end module
