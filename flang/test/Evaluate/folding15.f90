! RUN: %S/test_folding.sh %s %t %f18
! Test folding of array constructors with duplicate names for the implied
! DO variables
module m1
  integer, parameter :: expected(12) = [1, 2, 4, 6, 1, 2, 4, 6, 1, 2, 3, 6]
  integer, parameter :: dups(12) = &
    [ ((iDuplicate, iDuplicate = 1,j), &
       (2 * iDuplicate, iDuplicate = j,3 ), &
        j = 1,3 ) ]
  logical, parameter :: test_dups = all(dups == expected)
end module
