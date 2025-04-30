!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -c -Mstandard %s -v 2>&1 | grep flang1 | grep "\-standard"
! RUN: %flang -c %s -v 2>&1 | grep flang1 > %t
! RUN: not grep "\-standard" %t
! RUN: rm -f %t

