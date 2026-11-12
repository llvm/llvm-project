! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of COUNT()
module m
  logical, parameter :: arr(3,4) = reshape([(modulo(j, 2) == 1, j = 1, size(arr))], shape(arr))
  logical, parameter :: test_1 = count([1, 2, 3, 2, 1] < [(j, j=1, 5)]) == 2
  logical, parameter :: test_2 = count(arr) == 6
  logical, parameter :: test_3 = all(count(arr, dim=1) == [2, 1, 2, 1])
  logical, parameter :: test_4 = all(count(arr, dim=2) == [2, 2, 2])
  logical, parameter :: test_5 = count(logical(arr, kind=1)) == 6
  logical, parameter :: test_6 = count(logical(arr, kind=2)) == 6
end module
