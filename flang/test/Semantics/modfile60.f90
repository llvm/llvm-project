! RUN: %python %S/test_modfile.py %s %flang_fc1 -fbackslash
! Test Unicode escape sequences
module m
  integer, parameter :: wide = 4
  character(kind=wide, len=20), parameter :: ch = wide_"\u1234 \u56789abc"
  integer, parameter :: check(2) = [ iachar(ch(1:1)), iachar(ch(3:3)) ]
  logical, parameter :: valid = all(check == [int(z'1234'), int(z'56789abc')])
end

!Expect: m.mod
!module m
!integer(4),parameter::wide=4_4
!character(20_4,4),parameter::ch=4_"\341\210\264 \375\226\236\211\252\274                 "
!integer(4),parameter::check(1_8:2_8)=[INTEGER(4)::4660_4,1450744508_4]
!intrinsic::iachar
!logical(4),parameter::valid=.true._4
!intrinsic::all
!intrinsic::int
!end
