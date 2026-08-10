! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of PACK (valid cases)
module m
  integer, parameter :: arr(2,3) = reshape([1, 2, 3, 4, 5, 6], shape(arr))
  logical, parameter :: odds(*,*) = modulo(arr, 2) /= 0
  integer, parameter :: vect(*) = [(j, j=-10, -1)]
  logical, parameter :: test_pack_1 = all(pack(arr, .true.) == [arr])
  logical, parameter :: test_pack_2 = all(pack(arr, .false.) == [integer::])
  logical, parameter :: test_pack_3 = all(pack(arr, odds) == [1, 3, 5])
  logical, parameter :: test_pack_4 = all(pack(arr, .not. odds) == [2, 4, 6])
  logical, parameter :: test_pack_5 = all(pack(arr, .true., vect) == [1, 2, 3, 4, 5, 6, -4, -3, -2, -1])
  logical, parameter :: test_pack_6 = all(pack(arr, .false., vect) == vect)
  logical, parameter :: test_pack_7 = all(pack(arr, odds, vect) == [1, 3, 5, -7, -6, -5, -4, -3, -2, -1])
  logical, parameter :: test_pack_8 = all(pack(arr, .not. odds, vect) == [2, 4, 6, -7, -6, -5, -4, -3, -2, -1])
end module
