! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s --strict-whitespace --check-prefix=CHECK-E
! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK-OMP
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP

! Test in mixed way, i.e., combination of Fortran free source form
! and free source form with conditional compilation sentinel.
! CHECK-LABEL: subroutine mixed_form1()
! CHECK-E:{{^}}      i = 1 &
! CHECK-E:{{^}}!$    +100&
! CHECK-E:{{^}}!$    &+ 1000&
! CHECK-E:{{^}}      &+ 10 + 1&
! CHECK-E:{{^}}!$    & +100000&
! CHECK-E:{{^}}      &0000 + 1000000
! CHECK-OMP: i=1001001112_4
! CHECK-NO-OMP: i=1010011_4
subroutine mixed_form1()
   i = 1 &
  !$ +100&
  !$&+ 1000&
   &+ 10 + 1&
  !$& +100000&
   &0000 + 1000000
end subroutine

! Testing continuation lines in only Fortran Free form Source
! CHECK-LABEL: subroutine mixed_form2()
! CHECK-E:{{^}}      i = 1 +10 +100 + 1000 + 10000
! CHECK-OMP: i=11111_4
! CHECK-NO-OMP: i=11111_4
subroutine mixed_form2()
   i = 1 &
   +10 &
   &+100
   & + 1000 &
   + 10000
end subroutine

! Testing continuation line in only free source form conditional compilation sentinel.
! CHECK-LABEL: subroutine mixed_form3()
! CHECK-E:{{^}}!$    i=0
! CHECK-E:{{^}}!$    i = 1 &
! CHECK-E:{{^}}!$    & +10 &
! CHECK-E:{{^}}!$    &+100&
! CHECK-E:{{^}}!$    +1000
! CHECK-OMP: i=0_4
! CHECK-OMP: i=1111_4
! CHECK-NO-OMP-NOT: i=0_4
subroutine mixed_form3()
   !$ i=0
   !$ i = 1 &
   !$ & +10 &
   !$&+100&
   !$ +1000
end subroutine

! CHECK-LABEL: subroutine regression
! CHECK-E:{{^}}!$    real x, &
! CHECK-E:{{^}}      stop
! CHECK-OMP: REAL x, stop
! CHECK-NO-OMP-NOT: REAL x,
subroutine regression
!$ real x, &
 stop
end
