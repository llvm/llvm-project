!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

       ! RUN: %flang -c -Mextend %s
       ! RUN: %flang -c -ffixed-line-length-132 %s
       ! RUN: not %flang -c %s 2>&1 | FileCheck %s
       ! RUN: not %flang -c -ffixed-line-length-72 %s 2>&1 | FileCheck %s
       PRINT *, "1234567891234567891234567891234567891234567891234567891234567891234567891234567891" ! CHECK: Unmatched quote
       END

