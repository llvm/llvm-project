! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of SPREAD
module m1
  logical, parameter :: test_empty = size(spread(1, 1, 0)) == 0
  logical, parameter :: test_stov = all(spread(1, 1, 2) == [1, 1])
  logical, parameter :: test_vtom1 = all(spread([1, 2], 1, 3) == reshape([1, 1, 1, 2, 2, 2], [3, 2]))
  logical, parameter :: test_vtom2 = all(spread([1, 2], 2, 3) == reshape([1, 2, 1, 2, 1, 2], [2, 3]))
  logical, parameter :: test_vtom3 = all(spread([1, 2], 2, 3) == reshape([1, 2, 1, 2, 1, 2], [2, 3]))
  logical, parameter :: test_log1 = all(all(spread([.false., .true.], 1, 2), dim=2) .eqv. [.false., .false.])
  logical, parameter :: test_log2 = all(all(spread([.false., .true.], 2, 2), dim=2) .eqv. [.false., .true.])
  logical, parameter :: test_log3 = all(any(spread([.false., .true.], 1, 2), dim=2) .eqv. [.true., .true.])
  logical, parameter :: test_log4 = all(any(spread([.false., .true.], 2, 2), dim=2) .eqv. [.false., .true.])
end module
