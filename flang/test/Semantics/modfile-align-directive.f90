! RUN: %python %S/test_modfile.py %s %flang_fc1
! !DIR$ ALIGN is recorded in module files in its canonical spelling, after
! the declaration of the variable it applies to.
module m
  !dir$ align(64) :: a
  real :: a(10)
  !dir$ attributes align : 128 :: b
  real(8) :: b
  real :: c
  ! A member of COMMON storage keeps no alignment request: the directive is
  ! ignored with a warning and not written to the module file.
  !dir$ align(64) :: d
  real :: d
  common /blk/ d
end

!Expect: m.mod
!module m
!real(4)::a(1_8:10_8)
!!dir$ align(64) :: a
!real(8)::b
!!dir$ align(128) :: b
!real(4)::c
!real(4)::d
!common/blk/d
!end
