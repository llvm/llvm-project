! RUN: %python %S/test_modfile.py %s %flang_fc1
! REQUIRES: x86-registered-target
! Intrinsics SELECTED_INT_KIND, SELECTED_REAL_KIND, PRECISION, RANGE,
! RADIX, DIGITS

module m1
  ! REAL(KIND=10) handles 16 <= P < 19 (if available; ifort is KIND=16)
  integer, parameter :: realprec = precision(0._10)
  logical, parameter :: rpreccheck = 18 == realprec
  integer, parameter :: realpvals(*) = [16, 18]
  integer, parameter :: realpkinds(*) = &
    [(selected_real_kind(realpvals(j),0),j=1,size(realpvals))]
  logical, parameter :: realpcheck = all([10, 10] == realpkinds)
  ! REAL(KIND=10) handles 308 <= R < 4932 (if available; ifort is KIND=16)
  integer, parameter :: realrange = range(0._10)
  logical, parameter :: rrangecheck = 4931 == realrange
  integer, parameter :: realrvals(*) = [308, 4931]
  integer, parameter :: realrkinds(*) = &
    [(selected_real_kind(0,realrvals(j)),j=1,size(realrvals))]
  logical, parameter :: realrcheck = all([10, 10] == realrkinds)
  logical, parameter :: radixcheck = radix(0._10) == 2

  integer, parameter :: realdigits = digits(0._10)
  logical, parameter :: realdigitscheck = 64 == realdigits
end module m1
!Expect: m1.mod
!module m1
!integer(4),parameter::realprec=18_4
!intrinsic::precision
!logical(4),parameter::rpreccheck=.true._4
!integer(4),parameter::realpvals(1_8:*)=[INTEGER(4)::16_4,18_4]
!integer(4),parameter::realpkinds(1_8:*)=[INTEGER(4)::10_4,10_4]
!intrinsic::selected_real_kind
!intrinsic::size
!logical(4),parameter::realpcheck=.true._4
!intrinsic::all
!integer(4),parameter::realrange=4931_4
!intrinsic::range
!logical(4),parameter::rrangecheck=.true._4
!integer(4),parameter::realrvals(1_8:*)=[INTEGER(4)::308_4,4931_4]
!integer(4),parameter::realrkinds(1_8:*)=[INTEGER(4)::10_4,10_4]
!logical(4),parameter::realrcheck=.true._4
!logical(4),parameter::radixcheck=.true._4
!intrinsic::radix
!integer(4),parameter::realdigits=64_4
!intrinsic::digits
!logical(4),parameter::realdigitscheck=.true._4
!end
