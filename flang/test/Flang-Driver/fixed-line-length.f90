! Ensure argument -ffixed-line-length=n works as expected.

! REQUIRES: new-flang-driver

!--------------------------
! FLANG DRIVER (flang-new)
!--------------------------
! RUN: %flang-new -E %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=DEFAULTLENGTH
! RUN: not %flang-new -E -ffixed-line-length=-2 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=NEGATIVELENGTH
! RUN: not %flang-new -E -ffixed-line-length=3 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=INVALIDLENGTH
! RUN: %flang-new -E -ffixed-line-length=none %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang-new -E -ffixed-line-length=0 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang-new -E -ffixed-line-length=13 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=LENGTH13

!----------------------------------------
! FRONTEND FLANG DRIVER (flang-new -fc1)
!----------------------------------------
! RUN: %flang-new -fc1 -E %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=DEFAULTLENGTH
! RUN: not %flang-new -fc1 -E -ffixed-line-length=-2 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=NEGATIVELENGTH
! RUN: not %flang-new -fc1 -E -ffixed-line-length=3 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=INVALIDLENGTH
! RUN: %flang-new -fc1 -E -ffixed-line-length=none %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang-new -fc1 -E -ffixed-line-length=0 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=UNLIMITEDLENGTH
! RUN: %flang-new -fc1 -E -ffixed-line-length=13 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=LENGTH13

!-------------------------------------
! COMMAND ALIAS -ffixed-line-length-n
!-------------------------------------
! RUN: %flang-new -E -ffixed-line-length-13 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=LENGTH13
! RUN: %flang-new -fc1 -E -ffixed-line-length-13 %S/Inputs/fixed-line-length-test.f  2>&1 | FileCheck %s --check-prefix=LENGTH13

!-------------------------------------
! EXPECTED OUTPUT WITH DEFAULT LENGTH
!-------------------------------------
! The line should be trimmed to 72 characters when reading based on the default value of fixed line length.
! DEFAULTLENGTH: program{{(a{58})}}

!-----------------------------------------
! EXPECTED OUTPUT WITH A NEGATIVE LENGTH
!-----------------------------------------
! NEGATIVELENGTH: invalid value '-2' in 'ffixed-line-length=','value must be 'none' or a non-negative integer'

!-----------------------------------------
! EXPECTED OUTPUT WITH LENGTH LESS THAN 7
!-----------------------------------------
! INVALIDLENGTH: invalid value '3' in 'ffixed-line-length=','value must be at least seven'

!---------------------------------------
! EXPECTED OUTPUT WITH UNLIMITED LENGTH
!---------------------------------------
! The line should not be trimmed and so 73 characters (including spaces) should be read.
! UNLIMITEDLENGTH: program{{(a{59})}}

!--------------------------------
! EXPECTED OUTPUT WITH LENGTH 13
!--------------------------------
! LENGTH13: program
