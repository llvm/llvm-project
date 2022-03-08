! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of FINDLOC, MAXLOC, & MINLOC
module m1
  integer, parameter :: ia1(2:6) = [1, 2, 3, 2, 1]
  integer, parameter :: ia2(2:3,2:4) = reshape([1, 2, 3, 3, 2, 1], shape(ia2))
  integer, parameter :: ia3(2,0,2) = 0 ! middle dimension has zero extent

  logical, parameter :: test_fi1a = all(findloc(ia1, 1) == 1)
  logical, parameter :: test_fi1ar = rank(findloc(ia1, 1)) == 1
  logical, parameter :: test_fi1ak = kind(findloc(ia1, 1, kind=2)) == 2
  logical, parameter :: test_fi1ad = findloc(ia1, 1, dim=1) == 1
  logical, parameter :: test_fi1adr = rank(findloc(ia1, 1, dim=1)) == 0
  logical, parameter :: test_fi1b = all(findloc(ia1, 1, back=.true.) == 5)
  logical, parameter :: test_fi1c = all(findloc(ia1, 2, mask=[.false., .false., .true., .true., .true.]) == 4)

  logical, parameter :: test_fi2a = all(findloc(ia2, 1) == [1, 1])
  logical, parameter :: test_fi2ar = rank(findloc(ia2, 1)) == 1
  logical, parameter :: test_fi2b = all(findloc(ia2, 1, back=.true.) == [2, 3])
  logical, parameter :: test_fi2c = all(findloc(ia2, 2, mask=reshape([.false., .false., .true., .true., .true., .false.], shape(ia2))) == [1, 3])
  logical, parameter :: test_fi2d = all(findloc(ia2, 1, dim=1) == [1, 0, 2])
  logical, parameter :: test_fi2e = all(findloc(ia2, 1, dim=2) == [1, 3])

  logical, parameter :: test_xi1a = all(maxloc(ia1) == 3)
  logical, parameter :: test_xi1ar = rank(maxloc(ia1)) == 1
  logical, parameter :: test_xi1ak = kind(maxloc(ia1, kind=2)) == 2
  logical, parameter :: test_xi1ad = maxloc(ia1, dim=1) == 3
  logical, parameter :: test_xi1adr = rank(maxloc(ia1, dim=1)) == 0
  logical, parameter :: test_xi1b = all(maxloc(ia1, back=.true.) == 3)
  logical, parameter :: test_xi1c = all(maxloc(ia1, mask=[.false., .true., .false., .true., .true.]) == 2)
  logical, parameter :: test_xi1d = all(maxloc(ia1, mask=[.false., .true., .false., .true., .true.], back=.true.) == 4)

  logical, parameter :: test_xi2a = all(maxloc(ia2) == [1, 2])
  logical, parameter :: test_xi2ar = rank(maxloc(ia2)) == 1
  logical, parameter :: test_xi2b = all(maxloc(ia2, back=.true.) == [2, 2])
  logical, parameter :: test_xi2c = all(maxloc(ia2, mask=reshape([.false., .true., .true., .true., .true., .true.], shape(ia2))) == [1, 2])
  logical, parameter :: test_xi2d = all(maxloc(ia2, mask=reshape([.false., .true., .true., .true., .true., .true.], shape(ia2)), back=.true.) == [2, 2])
  logical, parameter :: test_xi2e = all(maxloc(ia2, dim=1) == [2, 1, 1])
  logical, parameter :: test_xi2f = all(maxloc(ia2, dim=1, back=.true.) == [2, 2, 1])
  logical, parameter :: test_xi2g = all(maxloc(ia2, dim=2) == [2, 2])

  logical, parameter :: test_ni1a = all(minloc(ia1) == 1)
  logical, parameter :: test_ni1ar = rank(minloc(ia1)) == 1
  logical, parameter :: test_ni1ak = kind(minloc(ia1, kind=2)) == 2
  logical, parameter :: test_ni1ad = minloc(ia1, dim=1) == 1
  logical, parameter :: test_ni1adr = rank(minloc(ia1, dim=1)) == 0
  logical, parameter :: test_ni1b = all(minloc(ia1, back=.true.) == 5)
  logical, parameter :: test_ni1c = all(minloc(ia1, mask=[.false., .true., .true., .true., .false.]) == 2)
  logical, parameter :: test_ni1d = all(minloc(ia1, mask=[.false., .true., .true., .true., .false.], back=.true.) == 4)

  logical, parameter :: test_ni2a = all(minloc(ia2) == [1, 1])
  logical, parameter :: test_ni2ar = rank(minloc(ia2)) == 1
  logical, parameter :: test_ni2b = all(minloc(ia2, back=.true.) == [2, 3])
  logical, parameter :: test_ni2c = all(minloc(ia2, mask=reshape([.false., .true., .true., .false., .true., .false.], shape(ia2))) == [2, 1])
  logical, parameter :: test_ni2d = all(minloc(ia2, mask=reshape([.false., .true., .true., .false., .true., .false.], shape(ia2)), back=.true.) == [1, 3])
  logical, parameter :: test_ni2e = all(minloc(ia2, dim=1) == [1, 1, 2])
  logical, parameter :: test_ni2f = all(minloc(ia2, dim=1, back=.true.) == [1, 2, 2])
  logical, parameter :: test_ni2g = all(minloc(ia2, dim=2) == [1, 3])

  logical, parameter :: test_xi3a = all(maxloc(ia3) == [0,0,0])
  logical, parameter :: test_xi3b = all(maxloc(ia3, back=.true.) == [0,0,0])
  logical, parameter :: test_xi3c = all(maxloc(ia3, dim=2) == reshape([0,0,0,0],shape=[2,2]))
  logical, parameter :: test_xi3d = all(maxloc(ia3, dim=2, back=.true.) == reshape([0,0,0,0],shape=[2,2]))
end module
