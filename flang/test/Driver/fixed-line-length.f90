! Ensure argument -ffixed-line-length=n works as expected.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -E -Xflang -fno-reformat %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=DEFAULTLENGTH
! RUN: not %flang -E -Xflang -fno-reformat -ffixed-line-length=-2 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=NEGATIVELENGTH
! RUN: not %flang -E -Xflang -fno-reformat -ffixed-line-length=3 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=INVALIDLENGTH
! RUN: %flang -E -Xflang -fno-reformat -ffixed-line-length=none %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang -E -Xflang -fno-reformat -ffixed-line-length=0 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang -E -Xflang -fno-reformat -ffixed-line-length=13 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=LENGTH13

!----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!----------------------------------------
! RUN: %flang_fc1 -E -fno-reformat %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=DEFAULTLENGTH
! RUN: not %flang_fc1 -E -fno-reformat -ffixed-line-length=-2 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=NEGATIVELENGTH
! RUN: not %flang_fc1 -E -fno-reformat -ffixed-line-length=3 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=INVALIDLENGTH
! RUN: %flang_fc1 -E -fno-reformat -ffixed-line-length=none %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang_fc1 -E -fno-reformat -ffixed-line-length=0 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang_fc1 -E -fno-reformat -ffixed-line-length=13 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=LENGTH13

!-------------------------------------
! COMMAND ALIAS -ffixed-line-length-n
!-------------------------------------
! RUN: %flang -E -Xflang -fno-reformat -ffixed-line-length-13 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=LENGTH13
! RUN: %flang_fc1 -E -fno-reformat -ffixed-line-length-13 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=LENGTH13

! The line should be trimmed to 72 characters when reading based on the default value of fixed line length.
! DEFAULTLENGTH: program{{(a{58})}}

! NEGATIVELENGTH: invalid value '-2' in 'ffixed-line-length=', value must be 'none' or a positive integer

! INVALIDLENGTH: invalid value '3' in 'ffixed-line-length=', value must be '7' or greater

! The line should not be trimmed and so 73 characters (including spaces) should be read.
! UNLIMITEDLENGTH: program{{(a{59})}}

! LENGTH13: program
