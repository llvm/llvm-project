! RUN: %python %S/test_modfile.py %s %flang_fc1
! Check legacy DEC structures
module m
  structure /s1/
    integer n/1/
    integer na(2)/2,3/
    structure /s1a/ m, ma(2)
      integer j/4/
      integer ja(2)/5,6/
    end structure
    structure m2(2), m3 ! anonymous
      integer k/7/
      integer %fill(3)
      integer ka(2)/8,9/
      real %fill(2)
    end structure
  end structure
  record/s1/ ra1, rb1
  record/s1a/ ra1a
  common/s1/ foo ! not a name conflict
  character*8 s1 ! not a name conflict
  integer t(2) /2*10/ ! DATA-like entity initialization
end

!Expect: m.mod
!module m
!structure /s1/
!integer(4)::n=1_4
!integer(4)::na(1_8:2_8)=[INTEGER(4)::2_4,3_4]
!structure /s1a/m,ma(1_8:2_8)
!integer(4)::j=4_4
!integer(4)::ja(1_8:2_8)=[INTEGER(4)::5_4,6_4]
!end structure
!structure m2(1_8:2_8),m3
!integer(4)::k=7_4
!integer(4)::%FILL(1_8:3_8)
!integer(4)::ka(1_8:2_8)=[INTEGER(4)::8_4,9_4]
!real(4)::%FILL(1_8:2_8)
!end structure
!end structure
!record/s1/::ra1
!record/s1/::rb1
!record/s1a/::ra1a
!real(4)::foo
!character(8_8,1)::s1
!integer(4)::t(1_8:2_8)
!common/s1/foo
!end
