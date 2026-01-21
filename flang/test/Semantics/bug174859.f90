! RUN: %python %S/test_modfile.py %s %flang_fc1
! Attributes of the derived type were also applied to type parameters.
module m
  type, abstract, public :: t(k)
    integer, kind :: k
  end type
end

!Expect: m.mod
!module m
!type,abstract::t(k)
!integer(4),kind::k
!end type
!end
