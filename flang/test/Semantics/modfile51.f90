! RUN: %python %S/test_modfile.py %s %flang_fc1
! Allow a generic spec that is not a name to be declared on an
! accessibility control statement
module m
  public :: assignment(=)
  public :: read(unformatted)
  public :: operator(.eq.)
  public :: operator(.smooth.)
end module

!Expect: m.mod
!module m
!interface assignment(=)
!end interface
!interface read(unformatted)
!end interface
!interface operator(.eq.)
!end interface
!interface operator(.smooth.)
!end interface
!end
