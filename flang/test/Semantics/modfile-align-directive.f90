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
  ! Neither does a member of EQUIVALENCE storage.
  !dir$ align(64) :: e1
  real :: e1, e2
  equivalence (e1, e2)
  ! Neither does an ALLOCATABLE or POINTER variable, which is not supported
  ! yet.
  !dir$ align(64) :: f, g
  real, allocatable :: f(:)
  real, pointer :: g(:)
end

!Expect: m.mod
!module m
!real(4)::a(1_8:10_8)
!!dir$ align(64) :: a
!real(8)::b
!!dir$ align(128) :: b
!real(4)::c
!real(4)::d
!real(4)::e1
!real(4)::e2
!real(4),allocatable::f(:)
!real(4),pointer::g(:)
!common/blk/d
!equivalence(e1,e2)
!end
