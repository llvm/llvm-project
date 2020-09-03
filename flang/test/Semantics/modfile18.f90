! RUN: %S/test_modfile.sh %s %t %f18
! Tests folding of array constructors

module m
  real, parameter :: a0 = 1.0_8
  real, parameter :: a1(2) = [real::2.0, 3.0]
  real, parameter :: a2(2) = [4.0, 5.0]
  real, parameter :: a3(0) = [real::]
  real, parameter :: a4(55) = [real::((1.0*k,k=1,j),j=1,10)]
  real, parameter :: a5(*) = [6.0, 7.0, 8.0]
  real, parameter :: a6(2) = [9, 10]
  real, parameter :: a7(6) = [([(1.0*k,k=1,j)],j=1,3)]
  real, parameter :: a8(13) = [real::1,2_1,3_2,4_4,5_8,6_16,7._2,8._3,9._4,10._8,11._16,(12.,12.5),(13._8,13.5)]
end module m

!Expect: m.mod
!module m
!real(4),parameter::a0=1._4
!real(4),parameter::a1(1_8:2_8)=[REAL(4)::2._4,3._4]
!real(4),parameter::a2(1_8:2_8)=[REAL(4)::4._4,5._4]
!real(4),parameter::a3(1_8:0_8)=[REAL(4)::]
!real(4),parameter::a4(1_8:55_8)=[REAL(4)::1._4,1._4,2._4,1._4,2._4,3._4,1._4,2._4,3._4,4._4,1._4,2._4,3._4,4._4,5._4,1._4,2._4,3._4,4._4,5._4,6._4,1._4,2._4,3._4,4._4,5._4,6._4,7._4,1._4,2._4,3._4,4._4,5._4,6._4,7._4,8._4,1._4,2._4,3._4,4._4,5._4,6._4,7._4,8._4,9._4,1._4,2._4,3._4,4._4,5._4,6._4,7._4,8._4,9._4,1.e1_4]
!real(4),parameter::a5(1_8:*)=[REAL(4)::6._4,7._4,8._4]
!real(4),parameter::a6(1_8:2_8)=[REAL(4)::9._4,1.e1_4]
!real(4),parameter::a7(1_8:6_8)=[REAL(4)::1._4,1._4,2._4,1._4,2._4,3._4]
!real(4),parameter::a8(1_8:13_8)=[REAL(4)::1._4,2._4,3._4,4._4,5._4,6._4,7._4,8._4,9._4,1.e1_4,1.1e1_4,1.2e1_4,1.3e1_4]
!end
