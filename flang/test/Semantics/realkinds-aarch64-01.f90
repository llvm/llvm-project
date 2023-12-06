! REQUIRES: aarch64-registered-target
! RUN: %python %S/test_modfile.py %s %flang_fc1 -triple aarch64-unknown-linux-gnu

module m1
  logical, parameter :: realpcheck = 16 == selected_real_kind(16)
end module m1
!Expect: m1.mod
!module m1
!logical(4),parameter::realpcheck=.true._4
!intrinsic::selected_real_kind
!end
