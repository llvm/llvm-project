! RUN: %python %S/test_modfile.py %s %flang_fc1
! The order of COMMON blocks in a module file follows source order, with
! blank COMMON last. More than 16 blocks are needed: below that, std::sort
! uses insertion sort, which can hide an invalid source-position comparator.
module m
  common /q/ q1
  common /c/ c1
  common /m/ m1
  common /t/ t1
  common /a/ a1
  common // blank1
  common /x/ x1
  common /e/ e1
  common /k/ k1
  common /r/ r1
  common /b/ b1
  common /z/ z1
  common /g/ g1
  common /n/ n1
  common /h/ h1
  common /w/ w1
  common /d/ d1
  common /s/ s1
  common /f/ f1
  common /p/ p1
  common /y/ y1
end

!Expect: m.mod
!module m
!  real(4)::q1
!  real(4)::c1
!  integer(4)::m1
!  real(4)::t1
!  real(4)::a1
!  real(4)::blank1
!  real(4)::x1
!  real(4)::e1
!  integer(4)::k1
!  real(4)::r1
!  real(4)::b1
!  real(4)::z1
!  real(4)::g1
!  integer(4)::n1
!  real(4)::h1
!  real(4)::w1
!  real(4)::d1
!  real(4)::s1
!  real(4)::f1
!  real(4)::p1
!  real(4)::y1
!  common/q/q1
!  common/c/c1
!  common/m/m1
!  common/t/t1
!  common/a/a1
!  common/x/x1
!  common/e/e1
!  common/k/k1
!  common/r/r1
!  common/b/b1
!  common/z/z1
!  common/g/g1
!  common/n/n1
!  common/h/h1
!  common/w/w1
!  common/d/d1
!  common/s/s1
!  common/f/f1
!  common/p/p1
!  common/y/y1
!  common//blank1
!end
