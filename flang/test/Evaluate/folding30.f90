! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of FINDLOC, MAXLOC, & MINLOC
module m1
  integer, parameter :: ia1(2:6) = [1, 2, 3, 2, 1]
  integer, parameter :: ia2(2:3,2:4) = reshape([1, 2, 3, 3, 2, 1], shape(ia2))

  logical, parameter :: fi1a = all(findloc(ia1, 1) == 1)
  logical, parameter :: fi1ar = rank(findloc(ia1, 1)) == 1
  logical, parameter :: fi1ak = kind(findloc(ia1, 1, kind=2)) == 2
  logical, parameter :: fi1ad = findloc(ia1, 1, dim=1) == 1
  logical, parameter :: fi1adr = rank(findloc(ia1, 1, dim=1)) == 0
  logical, parameter :: fi1b = all(findloc(ia1, 1, back=.true.) == 5)
  logical, parameter :: fi1c = all(findloc(ia1, 2, mask=[.false., .false., .true., .true., .true.]) == 4)

  logical, parameter :: fi2a = all(findloc(ia2, 1) == [1, 1])
  logical, parameter :: fi2ar = rank(findloc(ia2, 1)) == 1
  logical, parameter :: fi2b = all(findloc(ia2, 1, back=.true.) == [2, 3])
  logical, parameter :: fi2c = all(findloc(ia2, 2, mask=reshape([.false., .false., .true., .true., .true., .false.], shape(ia2))) == [1, 3])
  logical, parameter :: fi2d = all(findloc(ia2, 1, dim=1) == [1, 0, 2])
  logical, parameter :: fi2e = all(findloc(ia2, 1, dim=2) == [1, 3])

  logical, parameter :: xi1a = all(maxloc(ia1) == 3)
  logical, parameter :: xi1ar = rank(maxloc(ia1)) == 1
  logical, parameter :: xi1ak = kind(maxloc(ia1, kind=2)) == 2
  logical, parameter :: xi1ad = maxloc(ia1, dim=1) == 3
  logical, parameter :: xi1adr = rank(maxloc(ia1, dim=1)) == 0
  logical, parameter :: xi1b = all(maxloc(ia1, back=.true.) == 3)
  logical, parameter :: xi1c = all(maxloc(ia1, mask=[.false., .true., .false., .true., .true.]) == 2)
  logical, parameter :: xi1d = all(maxloc(ia1, mask=[.false., .true., .false., .true., .true.], back=.true.) == 4)

  logical, parameter :: xi2a = all(maxloc(ia2) == [1, 2])
  logical, parameter :: xi2ar = rank(maxloc(ia2)) == 1
  logical, parameter :: xi2b = all(maxloc(ia2, back=.true.) == [2, 2])
  logical, parameter :: xi2c = all(maxloc(ia2, mask=reshape([.false., .true., .true., .false., .true., .true.], shape(ia2))) == [2, 1])
  logical, parameter :: xi2d = all(maxloc(ia2, mask=reshape([.false., .true., .true., .false., .true., .true.], shape(ia2)), back=.true.) == [1, 3])
  logical, parameter :: xi2e = all(maxloc(ia2, dim=1) == [2, 1, 1])
  logical, parameter :: xi2f = all(maxloc(ia2, dim=1, back=.true.) == [2, 2, 1])
  logical, parameter :: xi2g = all(maxloc(ia2, dim=2) == [2, 2])

  logical, parameter :: ni1a = all(minloc(ia1) == 1)
  logical, parameter :: ni1ar = rank(minloc(ia1)) == 1
  logical, parameter :: ni1ak = kind(minloc(ia1, kind=2)) == 2
  logical, parameter :: ni1ad = minloc(ia1, dim=1) == 1
  logical, parameter :: ni1adr = rank(minloc(ia1, dim=1)) == 0
  logical, parameter :: ni1b = all(minloc(ia1, back=.true.) == 5)
  logical, parameter :: ni1c = all(minloc(ia1, mask=[.false., .true., .true., .true., .false.]) == 2)
  logical, parameter :: ni1d = all(minloc(ia1, mask=[.false., .true., .true., .true., .false.], back=.true.) == 4)

  logical, parameter :: ni2a = all(minloc(ia2) == [1, 1])
  logical, parameter :: ni2ar = rank(minloc(ia2)) == 1
  logical, parameter :: ni2b = all(minloc(ia2, back=.true.) == [2, 3])
  logical, parameter :: ni2c = all(minloc(ia2, mask=reshape([.false., .true., .true., .false., .true., .false.], shape(ia2))) == [2, 1])
  logical, parameter :: ni2d = all(minloc(ia2, mask=reshape([.false., .true., .true., .false., .true., .false.], shape(ia2)), back=.true.) == [1, 3])
  logical, parameter :: ni2e = all(minloc(ia2, dim=1) == [1, 1, 2])
  logical, parameter :: ni2f = all(minloc(ia2, dim=1, back=.true.) == [1, 2, 2])
  logical, parameter :: ni2g = all(minloc(ia2, dim=2) == [1, 3])
end module
