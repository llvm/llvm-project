! Test that -Mextend option is properly converted to -ffixed-line-length=132
! and that explicit -ffixed-line-length overrides -Mextend.

! RUN: %flang -### -Mextend %s 2>&1 | FileCheck %s
! RUN: %flang -### -Mextend -ffixed-line-length=80 %s 2>&1 | FileCheck %s --check-prefix=OVERRIDE
! RUN: %flang -### -ffixed-line-length=100 %s 2>&1 | FileCheck %s --check-prefix=EXPLICIT
! RUN: %flang -### -Mextend -ffixed-form %s 2>&1 | FileCheck %s --check-prefix=FIXEDFORM
! RUN: %flang -### -Mextend -ffree-form %s 2>&1 | FileCheck %s --check-prefix=FREEFORM

! CHECK: flang{{.*}}-fc1
! CHECK-SAME: -ffixed-line-length=132
! CHECK-NOT: -Mextend

! Test that explicit -ffixed-line-length=80 overrides -Mextend
! OVERRIDE: flang{{.*}}-fc1
! OVERRIDE-SAME: -ffixed-line-length=80
! OVERRIDE-NOT: -ffixed-line-length=132
! OVERRIDE-NOT: -Mextend

! Test that just -ffixed-line-length=100 works
! EXPLICIT: flang{{.*}}-fc1
! EXPLICIT-SAME: -ffixed-line-length=100
! EXPLICIT-NOT: -Mextend

! Test that -Mextend works with -ffixed-form
! FIXEDFORM: flang{{.*}}-fc1
! FIXEDFORM-SAME: -ffixed-line-length=132
! FIXEDFORM-SAME: -ffixed-form
! FIXEDFORM-NOT: -Mextend

! Test that -Mextend works with -ffree-form
! FREEFORM: flang{{.*}}-fc1
! FREEFORM-SAME: -ffixed-line-length=132
! FREEFORM-SAME: -ffree-form
! FREEFORM-NOT: -Mextend

! Dummy Fortran code to compile
program test_mextend
  implicit none
  integer :: i, j, k
  real :: x, y, z
  
  ! A long line to test extended line length
  ! 12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012
  x = 1.0
  y = 2.0
  z = x + y
  
  print *, "Test -Mextend option"
  print *, "x = ", x
  print *, "y = ", y
  print *, "z = ", z
end program test_mextend