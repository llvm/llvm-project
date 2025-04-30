!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -i8 -S -emit-llvm %s -o - 2>&1 | FileCheck %s
! RUN: %flang -fdefault-integer-8 -S -emit-llvm %s -o - 2>&1 | FileCheck %s

program i
  integer :: x ! CHECK: alloca i64
  logical :: c ! CHECK: alloca i64
  x = 5
  c = .true.
end program
