! RUN: %python %S/test_modfile.py %s %flang_fc1
! Use-association with VOLATILE or ASYNCHRONOUS

module m1
  real x
  integer y
  volatile z
contains
end

module m2
  use m1
  volatile x
  asynchronous y
end

!Expect: m1.mod
!module m1
!real(4)::x
!integer(4)::y
!real(4),volatile::z
!end

!Expect: m2.mod
!module m2
!use m1,only:x
!use m1,only:y
!use m1,only:z
!volatile::x
!asynchronous::y
!end
