! RUN: %flang_fc1 -I %S/Inputs/module-dir -fdebug-unparse-with-modules %s | FileCheck %s
module m1
  use iso_fortran_env
  use BasicTestModuleTwo
  implicit none
  type(t2) y
  real(real32) x
end

program test
  use m1
  use BasicTestModuleTwo
  implicit none
  x = 123.
  y = t2()
end

!CHECK-NOT: module iso_fortran_env
!CHECK: module basictestmoduletwo
!CHECK: type::t2
!CHECK: end type
!CHECK: end
!CHECK: module m1
!CHECK:  use :: iso_fortran_env
!CHECK:  implicit none
!CHECK:  real(kind=real32) x
!CHECK: end module
!CHECK: program test
!CHECK:  use :: m1
!CHECK:  use :: basictestmoduletwo
!CHECK:  implicit none
!CHECK:  x = 123.
!CHECK:  y = t2()
!CHECK: end program
