! RUN: %S/test_modfile.sh %s %f18 %t
module m
  logical b
  bind(C) :: /cb2/
  common //t
  common /cb/ x(2:10) /cb2/a,b,c
  common /cb/ y,z
  common w
  common u,v
  complex w
  dimension b(4,4)
  bind(C, name="CB") /cb/
  common /b/ cb
end

!Expect: m.mod
!module m
!  logical(4)::b(1_8:4_8,1_8:4_8)
!  real(4)::t
!  real(4)::x(2_8:10_8)
!  real(4)::a
!  real(4)::c
!  real(4)::y
!  real(4)::z
!  real(4)::u
!  real(4)::v
!  complex(4)::w
!  real(4)::cb
!  common/cb2/a,b,c
!  bind(c)::/cb2/
!  common//t,w,u,v
!  common/cb/x,y,z
!  bind(c, name="CB")::/cb/
!  common/b/cb
!end
