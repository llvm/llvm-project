! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test 7.6 enum values

module m1
  integer, parameter :: x(1) = [4]
  enum, bind(C)
    enumerator :: red, green
    enumerator blue
    enumerator yellow
    enumerator :: purple = 2
    enumerator :: brown
  end enum

  enum, bind(C)
    enumerator :: oak, beech = -rank(x)*x(1), pine, poplar = brown
  end enum

  ! F2023 7.6.1 errata f23/013: BOZ enumerator initializers are
  ! interpreted as INT(boz, C_INT), and following enumerators increment.
  enum, bind(C)
    enumerator :: boz = z'2a', after_boz
  end enum

end

!Expect: m1.mod
!module m1
!integer(4),parameter::x(1_8:1_8)=[INTEGER(4)::4_4]
!integer(4),parameter::red=0_4
!integer(4),parameter::green=1_4
!integer(4),parameter::blue=2_4
!integer(4),parameter::yellow=3_4
!integer(4),parameter::purple=2_4
!integer(4),parameter::brown=3_4
!integer(4),parameter::oak=0_4
!integer(4),parameter::beech=-4_4
!intrinsic::rank
!integer(4),parameter::pine=-3_4
!integer(4),parameter::poplar=3_4
!integer(4),parameter::boz=42_4
!integer(4),parameter::after_boz=43_4
!end
