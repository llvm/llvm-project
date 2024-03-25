! RUN: %python %S/test_modfile.py %s %flang_fc1
! Ensure that a module can be forward-referenced within a compilation unit.
module m1
  use m2
end

module m2
  use m3
end

module m3
  integer n
end

!Expect: m1.mod
!module m1
!use m2,only:n
!end

!Expect: m2.mod
!module m2
!use m3,only:n
!end

!Expect: m3.mod
!module m3
!integer(4)::n
!end
