!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -c -Mpreprocess %s
! RUN: %flang -c -cpp %s
! RUN: not %flang -c %s 2>&1 | FileCheck %s

! CHECK: Label field of continuation line is not blank
#define MESSAGE "Hello world"

program cpp
  print *, MESSAGE
end program

