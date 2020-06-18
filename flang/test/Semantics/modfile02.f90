! RUN: %S/test_modfile.sh %s %t %f18
! Check modfile generation for private type in public API.

module m
  type, private :: t1
    integer :: i
  end type
  type, private :: t2
    integer :: i
  end type
  type(t1) :: x1
  type(t2), private :: x2
end

!Expect: m.mod
!module m
!type,private::t1
!integer(4)::i
!end type
!type,private::t2
!integer(4)::i
!end type
!type(t1)::x1
!type(t2),private::x2
!end
