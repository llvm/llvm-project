! RUN: %python %S/test_errors.py %s %flang_fc1
! Test USE statements that use same module multiple times mixed with rename
! clauses and ONLY clauses
module m1
  integer :: a = 1
  integer :: b = 2
end module m1
module m2
  integer :: a = 3
end module m2
module m3
  integer :: a = 1
  type t1
    real t1_value
  end type
  type t2
    complex t2_value
  end type
end module m3
module m4
  use m1
end module m4
module m5
  use m1
  use m1, z=>a
end module m5
module m6
  use m1, only : a
end module m6
program testUse1
  use m1
  use m1,z=>a ! This prevents the use association of m1's "a" as local "a"
  use m2 ! m2's version of "a" gets use associated
  !ERROR: 'a' is use-associated from module 'm2' and cannot be re-declared
  integer :: a = 2
end
subroutine testUse2
  use m1,only : a ! This forces the use association of m1's "a" as local "a"
  use m1,z=>a ! This rename doesn't affect the previous forced USE association
  !ERROR: 'a' is use-associated from module 'm1' and cannot be re-declared
  integer :: a = 2
end
subroutine testUse3
  use m1 ! By itself, this would use associate m1's "a" with a local "a"
  use m1,z=>a ! This rename of m1'a "a" removes the previous use association
  integer :: a = 2
end
subroutine testUse4
  use m1,only : a ! Use associate m1's "a" with local "a"
  use m1,z=>a ! Also use associate m1's "a" with local "z", also pulls in "b"
  !ERROR: 'b' is use-associated from module 'm1' and cannot be re-declared
  integer :: b = 2
end
subroutine testUse5
  use m1,z=>a ! The rename prevents creation of a local "a"
  use m1 ! Does not create a local "a" because of the previous rename
  integer :: a = 2
end
subroutine testUse6
  use m1, z => a ! Hides m1's "a"
  use m1, y => b ! Hides m1's "b"
  integer :: a = 4 ! OK
  integer :: b = 5 ! OK
end
subroutine testUse7
  use m3,t1=>t2,t2=>t1 ! Looks weird but all is good
  type(t1) x
  type(t2) y
  x%t2_value = a
  y%t1_value = z
end
subroutine testUse8
  use m4 ! This USE associates all of m1
  !ERROR: 'a' is use-associated from module 'm4' and cannot be re-declared
  integer :: a = 2
end
subroutine testUse9
  use m5
  integer :: a = 2
end
subroutine testUse10
  use m4
  use m4, z=>a ! This rename erases the USE assocated "a" from m1
  integer :: a = 2
end
subroutine testUse11
  use m6
  use m6, z=>a ! This rename erases the USE assocated "a" from m1
  integer :: a = 2
end
subroutine testUse12
  use m4 ! This USE associates "a" from m1
  use m1, z=>a ! This renames the "a" from m1, but not the one through m4
  !ERROR: 'a' is use-associated from module 'm4' and cannot be re-declared
  integer :: a = 2
end
