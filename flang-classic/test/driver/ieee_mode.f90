!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -c -Knoieee %s -v 2>&1 | grep flang1 | grep "-ieee 0"
! RUN: %flang -c -Kieee %s -v 2>&1 | grep flang1 | grep "-ieee 1"
! RUN: %flang -c %s -v 2>&1 | grep flang1 | grep "-ieee 1"

! RUN: %flang -c -Knoieee %s -v 2>&1 | grep flang2 | grep "-ieee 0"
! RUN: %flang -c -Kieee %s -v 2>&1 | grep flang2 | grep "-ieee 1"
! RUN: %flang -c %s -v 2>&1 | grep flang2 | grep "-ieee 1"
