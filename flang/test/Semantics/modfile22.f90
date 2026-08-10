! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test character length conversions in constructors

module m
type :: t(k)
  integer, kind :: k = 1
  character(kind=k,len=1) :: a
  character(kind=k,len=3) :: b
end type t
type(t), parameter :: p = t(k=1)(a='xx',b='xx')
character(len=2), parameter :: c2(3) = [character(len=2) :: 'x', 'xx', 'xxx']
end module m

!Expect: m.mod
!module m
!type::t(k)
!integer(4),kind::k=1_4
!character(1_4,k)::a
!character(3_4,k)::b
!end type
!type(t(k=1_4)),parameter::p=t(k=1_4)(a="x",b="xx ")
!character(2_4,1),parameter::c2(1_8:3_8)=[CHARACTER(KIND=1,LEN=2)::"x ","xx","xx"]
!end
