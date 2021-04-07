!-----------
! RUN lines
!-----------
! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s --check-prefix=DEFINED
! RUN: %flang_fc1 -E -cpp %s 2>&1 | FileCheck %s --check-prefix=DEFINED
! RUN: %flang_fc1 -E -nocpp %s 2>&1 | FileCheck %s --check-prefix=NOT_DEFINED

!-----------------
! EXPECTED OUTPUT
!-----------------
! DEFINED: flang = 1
! DEFINED-NEXT: flang_major = {{[1-9][0-9]*$}}

! NOT_DEFINED: flang = __flang__
! NOT_DEFINED-NEXT: flang_major = __flang_major__

integer, parameter :: flang = __flang__
integer, parameter :: flang_major = __flang_major__
