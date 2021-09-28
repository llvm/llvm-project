! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of FINDLOC
module m1
  integer, parameter :: ia1(2:6) = [1, 2, 3, 2, 1]
  integer, parameter :: ia2(2:3,2:4) = reshape([1, 2, 3, 3, 2, 1], shape(ia2))

  logical, parameter :: ti1a = all(findloc(ia1, 1) == 1)
  logical, parameter :: ti1ar = rank(findloc(ia1, 1)) == 1
  logical, parameter :: ti1ak = kind(findloc(ia1, 1, kind=2)) == 2
  logical, parameter :: ti1ad = findloc(ia1, 1, dim=1) == 1
  logical, parameter :: ti1adr = rank(findloc(ia1, 1, dim=1)) == 0
  logical, parameter :: ti1b = all(findloc(ia1, 1, back=.true.) == 5)
  logical, parameter :: ti1c = all(findloc(ia1, 2, mask=[.false., .false., .true., .true., .true.]) == 4)

  logical, parameter :: ti2a = all(findloc(ia2, 1) == [1, 1])
  logical, parameter :: ti2ar = rank(findloc(ia2, 1)) == 1
  logical, parameter :: ti2b = all(findloc(ia2, 1, back=.true.) == [2, 3])
  logical, parameter :: ti2c = all(findloc(ia2, 2, mask=reshape([.false., .false., .true., .true., .true., .false.], shape(ia2))) == [1, 3])
  logical, parameter :: ti2d = all(findloc(ia2, 1, dim=1) == [1, 0, 2])
  logical, parameter :: ti2e = all(findloc(ia2, 1, dim=2) == [1, 3])
end module
