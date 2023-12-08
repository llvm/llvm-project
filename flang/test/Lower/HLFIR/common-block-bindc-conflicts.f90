! Test the mixing BIND(C) and non BIND(C) common blocks.

! RUN: %flang_fc1 -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=UNDERSCORING
! RUN: %flang_fc1 -emit-llvm -fno-underscoring %s -o - 2>&1 | FileCheck %s --check-prefix=NO-UNDERSCORING

! Scenario 1: Fortran symbols collide, but not the object file names, emit different
! globals for each common
subroutine bindc_common_with_same_fortran_name()
  real :: x
  common /com1/ x
  bind(c, name="not_com1") :: /com1/
  print *, x
end subroutine

subroutine bindc_common_with_same_fortran_name_2()
  real :: x(2), y(2)
  common /com1/ x
  print *, x
end subroutine

! Scenario 2: object file names of common block may collide (depending on
! underscoring option). Merge common block into a single global symbol.
subroutine bindc_common_colliding_with_normal_common()
  real :: x, y
  common /com3/ x
  common /com4/ y
  bind(c, name="some_common_") :: /com3/
  bind(c, name="__BLNK__") :: /com4/
  print *, x, y
end subroutine
subroutine bindc_common_colliding_with_normal_common_2()
  real :: x(2), y(2)
  common /some_common/ x
  common // y
  print *, x, y
end subroutine

! UNDERSCORING: @__BLNK__ = common global [8 x i8] zeroinitializer
! UNDERSCORING: @com1_ = common global [8 x i8] zeroinitializer
! UNDERSCORING: @not_com1 = common global [4 x i8] zeroinitializer
! UNDERSCORING: @some_common_ = common global [8 x i8] zeroinitializer

! NO-UNDERSCORING: @__BLNK__ = common global [8 x i8] zeroinitializer
! NO-UNDERSCORING: @com1 = common global [8 x i8] zeroinitializer
! NO-UNDERSCORING: @not_com1 = common global [4 x i8] zeroinitializer
! NO-UNDERSCORING: @some_common = common global [8 x i8] zeroinitializer
! NO-UNDERSCORING: @some_common_ = common global [4 x i8] zeroinitializer
