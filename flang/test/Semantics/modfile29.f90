! RUN: %python %S/test_modfile.py %s %flang_fc1
! Check that implicitly typed entities get a type in the module file.

module m
  public :: a
  private :: b
  protected :: i
  allocatable :: j
end

!Expect: m.mod
!module m
! real(4)::a
! real(4),private::b
! integer(4),protected::i
! integer(4),allocatable::j
!end
