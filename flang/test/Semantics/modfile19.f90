! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
  implicit complex(8)(z)
  real :: x
  namelist /nl1/ x, y
  namelist /nl2/ y, x
  namelist /nl1/ i, z
  complex(8) :: z
  real :: y
end

!Expect: m.mod
!module m
!  real(4)::x
!  integer(4)::i
!  complex(8)::z
!  real(4)::y
!  namelist/nl1/x,y,i,z
!  namelist/nl2/y,x
!end
