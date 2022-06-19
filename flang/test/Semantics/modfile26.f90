! RUN: %python %S/test_modfile.py %s %flang_fc1
! Intrinsics SELECTED_INT_KIND, SELECTED_REAL_KIND, PRECISION, RANGE,
! RADIX, DIGITS

module m1
  ! INTEGER(KIND=1)  handles  0 <= P < 3
  ! INTEGER(KIND=2)  handles  3 <= P < 5
  ! INTEGER(KIND=4)  handles  5 <= P < 10
  ! INTEGER(KIND=8)  handles 10 <= P < 19
  ! INTEGER(KIND=16) handles 19 <= P < 39
  integer, parameter :: iranges(*) = &
    [range(0_1), range(0_2), range(0_4), range(0_8), range(0_16)]
  logical, parameter :: ircheck = all([2, 4, 9, 18, 38] == iranges)
  integer, parameter :: intpvals(*) = [0, 2, 3, 4, 5, 9, 10, 18, 19, 38, 39]
  integer, parameter :: intpkinds(*) = &
    [(selected_int_kind(intpvals(j)),j=1,size(intpvals))]
  logical, parameter :: ipcheck = &
    all([1, 1, 2, 2, 4, 4, 8, 8, 16, 16, -1] == intpkinds)

  ! REAL(KIND=2)  handles  0 <= P < 4  (if available)
  ! REAL(KIND=3)  handles  0 <= P < 3  (if available)
  ! REAL(KIND=4)  handles  4 <= P < 7
  ! REAL(KIND=8)  handles  7 <= P < 16
  ! REAL(KIND=10) handles 16 <= P < 19 (if available; ifort is KIND=16)
  ! REAL(KIND=16) handles 19 <= P < 34 (would be 32 with Power double/double)
  integer, parameter :: realprecs(*) = &
    [precision(0._2), precision(0._3), precision(0._4), precision(0._8), &
     precision(0._10), precision(0._16)]
  logical, parameter :: rpreccheck = all([3, 2, 6, 15, 18, 33] == realprecs)
  integer, parameter :: realpvals(*) = [0, 3, 4, 6, 7, 15, 16, 18, 19, 33, 34]
  integer, parameter :: realpkinds(*) = &
    [(selected_real_kind(realpvals(j),0),j=1,size(realpvals))]
  logical, parameter :: realpcheck = &
    all([2, 2, 4, 4, 8, 8, 10, 10, 16, 16, -1] == realpkinds)
  ! REAL(KIND=2)  handles  0 <= R < 5 (if available)
  ! REAL(KIND=3)  handles  5 <= R < 38 (if available, same range as KIND=4)
  ! REAL(KIND=4)  handles  5 <= R < 38 (if no KIND=3)
  ! REAL(KIND=8)  handles 38 <= R < 308
  ! REAL(KIND=10) handles 308 <= R < 4932 (if available; ifort is KIND=16)
  ! REAL(KIND=16) handles 308 <= R < 4932 (except Power double/double)
  integer, parameter :: realranges(*) = &
    [range(0._2), range(0._3), range(0._4), range(0._8), range(0._10), &
     range(0._16)]
  logical, parameter :: rrangecheck = &
    all([4, 37, 37, 307, 4931, 4931] == realranges)
  integer, parameter :: realrvals(*) = &
    [0, 4, 5, 37, 38, 307, 308, 4931, 4932]
  integer, parameter :: realrkinds(*) = &
    [(selected_real_kind(0,realrvals(j)),j=1,size(realrvals))]
  logical, parameter :: realrcheck = &
    all([2, 2, 3, 3, 8, 8, 10, 10, -2] == realrkinds)
  logical, parameter :: radixcheck = &
    all([radix(0._2), radix(0._3), radix(0._4), radix(0._8), &
         radix(0._10), radix(0._16)] == 2)
  integer, parameter :: intdigits(*) = &
    [digits(0_1), digits(0_2), digits(0_4), digits(0_8), digits(0_16)]
  logical, parameter :: intdigitscheck = &
    all([7, 15, 31, 63, 127] == intdigits)
  integer, parameter :: realdigits(*) = &
    [digits(0._2), digits(0._3), digits(0._4), digits(0._8), digits(0._10), &
     digits(0._16)]
  logical, parameter :: realdigitscheck = &
    all([11, 8, 24, 53, 64, 113] == realdigits)
end module m1
!Expect: m1.mod
!module m1
!integer(4),parameter::iranges(1_8:*)=[INTEGER(4)::2_4,4_4,9_4,18_4,38_4]
!intrinsic::range
!logical(4),parameter::ircheck=.true._4
!intrinsic::all
!integer(4),parameter::intpvals(1_8:*)=[INTEGER(4)::0_4,2_4,3_4,4_4,5_4,9_4,10_4,18_4,19_4,38_4,39_4]
!integer(4),parameter::intpkinds(1_8:*)=[INTEGER(4)::1_4,1_4,2_4,2_4,4_4,4_4,8_4,8_4,16_4,16_4,-1_4]
!intrinsic::selected_int_kind
!intrinsic::size
!logical(4),parameter::ipcheck=.true._4
!integer(4),parameter::realprecs(1_8:*)=[INTEGER(4)::3_4,2_4,6_4,15_4,18_4,33_4]
!intrinsic::precision
!logical(4),parameter::rpreccheck=.true._4
!integer(4),parameter::realpvals(1_8:*)=[INTEGER(4)::0_4,3_4,4_4,6_4,7_4,15_4,16_4,18_4,19_4,33_4,34_4]
!integer(4),parameter::realpkinds(1_8:*)=[INTEGER(4)::2_4,2_4,4_4,4_4,8_4,8_4,10_4,10_4,16_4,16_4,-1_4]
!intrinsic::selected_real_kind
!logical(4),parameter::realpcheck=.true._4
!integer(4),parameter::realranges(1_8:*)=[INTEGER(4)::4_4,37_4,37_4,307_4,4931_4,4931_4]
!logical(4),parameter::rrangecheck=.true._4
!integer(4),parameter::realrvals(1_8:*)=[INTEGER(4)::0_4,4_4,5_4,37_4,38_4,307_4,308_4,4931_4,4932_4]
!integer(4),parameter::realrkinds(1_8:*)=[INTEGER(4)::2_4,2_4,3_4,3_4,8_4,8_4,10_4,10_4,-2_4]
!logical(4),parameter::realrcheck=.true._4
!logical(4),parameter::radixcheck=.true._4
!intrinsic::radix
!integer(4),parameter::intdigits(1_8:*)=[INTEGER(4)::7_4,15_4,31_4,63_4,127_4]
!intrinsic::digits
!logical(4),parameter::intdigitscheck=.true._4
!integer(4),parameter::realdigits(1_8:*)=[INTEGER(4)::11_4,8_4,24_4,53_4,64_4,113_4]
!logical(4),parameter::realdigitscheck=.true._4
!end
