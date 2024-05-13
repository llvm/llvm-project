! RUN: %python %S/test_modfile.py %s %flang_fc1
! Ensure proper formatting of component initializers in PDTs;
! they should be unparsed from their parse trees.
module m
  type :: t(k)
    integer, kind :: k
    real(kind=k) :: x = real(0., kind=k)
  end type
end module

!Expect: m.mod
!module m
!type::t(k)
!integer(4),kind::k
!real(int(int(k,kind=4),kind=8))::x=real(0., kind=k)
!end type
!intrinsic::real
!end
