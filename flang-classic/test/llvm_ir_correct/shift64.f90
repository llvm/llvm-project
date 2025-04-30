!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -S -emit-llvm %s -o - 2>&1 | FileCheck %s

program shift64
  integer(8) :: x, y

  ! CHECK: sext i32 3 to i64
  ! CHECK-NEXT: shl
  x = ishft(x, 3)
  ! CHECK: sext i32 5 to i64
  ! CHECK-NEXT: lshr
  x = ishft(x, -5)
  ! CHECK: @ftn_i_kishft
  y = ishft(x, y)
end program
