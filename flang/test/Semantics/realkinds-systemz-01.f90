! REQUIRES: systemz-registered-target
! REQUIRES: flang-supports-f128-math
! RUN: %python %S/test_modfile.py %s %flang_fc1 -triple s390x-unknown-linux-gnu

module m1
  logical, parameter :: realpcheck = 16 == selected_real_kind(16)
end module m1
!Expect: m1.mod
!module m1
!logical(4),parameter::realpcheck=.true._4
!intrinsic::selected_real_kind
!end
