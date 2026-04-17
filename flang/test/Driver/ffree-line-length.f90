! Ensure argument -ffree-line-length=n works as expected.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -E -Xflang -fno-reformat %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: not %flang -E -Xflang -fno-reformat -ffree-line-length=-2 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=NEGATIVELENGTH
! RUN: not %flang -E -Xflang -fno-reformat -ffree-line-length=3 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=INVALIDLENGTH
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length=none %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length=0 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length=13 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=LENGTH13

!----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!----------------------------------------
! RUN: %flang_fc1 -E -fno-reformat %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: not %flang_fc1 -E -fno-reformat -ffree-line-length=-2 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=NEGATIVELENGTH
! RUN: not %flang_fc1 -E -fno-reformat -ffree-line-length=3 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=INVALIDLENGTH
! RUN: %flang_fc1 -E -fno-reformat -ffree-line-length=none %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang_fc1 -E -fno-reformat -ffree-line-length=0 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang_fc1 -E -fno-reformat -ffree-line-length=13 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=LENGTH13

!-------------------------------------
! COMMAND ALIAS -ffree-line-length-n
!-------------------------------------
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length-13 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=LENGTH13
! RUN: %flang_fc1 -E -fno-reformat -ffree-line-length-13 %S/Inputs/free-line-length-test.f90  2>&1 | FileCheck %s --check-prefix=LENGTH13


! NEGATIVELENGTH: invalid value '-2' in 'ffree-line-length=', value must be 'none' or a positive integer

! INVALIDLENGTH: invalid value '3' in 'ffree-line-length=', value must be '7' or greater

! The line should not be trimmed and should be read.
! UNLIMITEDLENGTH: program {{(a{118})}}

! LENGTH13: program

