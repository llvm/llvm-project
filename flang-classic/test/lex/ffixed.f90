!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -c %s
! RUN: %flang -c -ffree-form %s
! RUN: %flang -c -fno-fixed-form %s
! RUN: %flang -c -Mfree %s
! RUN: %flang -c -Mfreeform %s
! RUN: not %flang -c -ffixed-form %s 2>&1 | FileCheck %s
! RUN: not %flang -c -fno-free-form %s 2>&1 | FileCheck %s
! RUN: not %flang -c -Mfixed %s 2>&1 | FileCheck %s
! RUN: not %flang -c -Mnofree %s 2>&1 | FileCheck %s
! RUN: not %flang -c -Mnofreeform %s 2>&1 | FileCheck %s

program f ! CHECK: Label field of continuation line is not blank
end program ! CHECK: Label field of continuation line is not blank
