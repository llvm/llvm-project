! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of ISHFTC
module m
  integer, parameter :: shift8s(*) = ishftc(257, shift = [(ict, ict = -8, 8)], size=8)
  integer, parameter :: expect1(*) = 256 + [1, 2, 4, 8, 16, 32, 64, 128, &
                                            1, 2, 4, 8, 16, 32, 64, 128, 1]
  logical, parameter :: test_1 = all(shift8s == expect1)
  integer, parameter :: sizes(*) = [(ishftc(257, ict, [(isz, isz = 1, 8)]), ict = -1, 1)]
  integer, parameter :: expect2(*) = 256 + [[1, 2, 4, 8, 16, 32, 64, 128], &
                                            [(1, j = 1, 8)], &
                                            [1, (2, j = 2, 8)]]
  logical, parameter :: test_2 = all(sizes == expect2)
end module

