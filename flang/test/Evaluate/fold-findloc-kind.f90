! RUN: %python %S/test_folding.py %s %flang_fc1
! FINDLOC on non-default LOGICAL kinds must compare with .EQV. at that
! kind. Folding the comparison as LOGICAL(4) drops the result.
module m1
  logical(1), parameter :: la1(*) = [.false._1, .true._1, .false._1, .true._1]
  logical(8), parameter :: la8(*) = [.true._8, .false._8, .true._8]
  logical, parameter :: test_fl1 = all(findloc(la1, .true._1) == 2)
  logical, parameter :: test_fl1b = all(findloc(la1, .true._1, back=.true.) == 4)
  logical, parameter :: test_fl1m = findloc(la1, .false._1, dim=1) == 1
  logical, parameter :: test_fl8 = all(findloc(la8, .true._8) == 1)
  logical, parameter :: test_fl8b = all(findloc(la8, .false._8, back=.true.) == 2)
  logical, parameter :: test_fl8m = all(findloc(la8, .false._8) == 2)
end module
