! Ensure argument -ffree-line-length=n works as expected.

!--------------------------
! FLANG DRIVER (flang)
!--------------------------
! RUN: %flang -E -Xflang -fno-reformat %s  2>&1 | FileCheck %s --check-prefix=LENGTH35
! RUN: not %flang -E -Xflang -fno-reformat -ffree-line-length=-2 %s  2>&1 | FileCheck %s --check-prefix=NEGATIVELENGTH
! RUN: not %flang -E -Xflang -fno-reformat -ffree-line-length=abcd %s  2>&1 | FileCheck %s --check-prefix=INVALIDLENGTH
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length=none %s  2>&1 | FileCheck %s --check-prefix=LENGTH35
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length=0 %s  2>&1 | FileCheck %s --check-prefix=LENGTH35
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length=13 %s  2>&1 | FileCheck %s --check-prefix=LENGTH13

!----------------------------------------
! FRONTEND FLANG DRIVER (flang -fc1)
!----------------------------------------
! RUN: %flang_fc1 -E -fno-reformat %s  2>&1 | FileCheck %s --check-prefix=LENGTH35
! RUN: not %flang_fc1 -E -fno-reformat -ffree-line-length=-2 %s  2>&1 | FileCheck %s --check-prefix=NEGATIVELENGTH
! RUN: not %flang_fc1 -E -fno-reformat -ffree-line-length-abcd %s  2>&1 | FileCheck %s --check-prefix=INVALIDLENGTH
! RUN: %flang_fc1 -E -fno-reformat -ffree-line-length=none %s  2>&1 | FileCheck %s --check-prefix=LENGTH35
! RUN: %flang_fc1 -E -fno-reformat -ffree-line-length=0 %s  2>&1 | FileCheck %s --check-prefix=LENGTH35
! RUN: %flang_fc1 -E -fno-reformat -ffree-line-length=13 %s  2>&1 | FileCheck %s --check-prefix=LENGTH13

!-------------------------------------
! COMMAND ALIAS -ffree-line-length-n
!-------------------------------------
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length-13 %s  2>&1 | FileCheck %s --check-prefix=LENGTH13
! RUN: %flang_fc1 -E -fno-reformat -ffree-line-length-13 %s  2>&1 | FileCheck %s --check-prefix=LENGTH13

!-------------------------------------
! BOTH FLAGS
!-------------------------------------
! RUN: %flang -E -Xflang -fno-reformat -ffixed-line-length=13 -ffree-line-length=500 %s  2>&1 | FileCheck %s --check-prefix=LENGTH35
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length=500 -ffixed-line-length=13 %s  2>&1 | FileCheck %s --check-prefix=LENGTH35
! RUN: %flang -E -Xflang -fno-reformat -ffixed-line-length=100 -ffree-line-length=13 %s  2>&1 | FileCheck %s --check-prefix=LENGTH13
! RUN: %flang -E -Xflang -fno-reformat -ffree-line-length=13 -ffixed-line-length=100 %s  2>&1 | FileCheck %s --check-prefix=LENGTH13

! The length of the line below is exactly 35 characters
program arbitrary_program_test_name 
end

! NEGATIVELENGTH: invalid value '-2' in 'ffree-line-length=', value must be 'none' or a positive integer

! INVALIDLENGTH: invalid value 'abcd' in 'ffree-line-length=', value must be 'none' or a positive integer

! The line should not be trimmed and should be read.
! LENGTH35: program arbitrary_program_test_name

! LENGTH13: program arbit
