!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! Test fix for kind mismatch when intrinsic len is used

! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CONVERT
! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s -check-prefix=NO_CONVERT
! RUN: %flang -fdefault-integer-8 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CONVERT_I8
! RUN: %flang -fdefault-integer-8 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=NO_CONVERT_I8
subroutine s1(x)
  character(len=x) :: s1_str
  integer :: s1_test_len
  s1_test_len = len(s1_str)
  print *, s1_test_len
end subroutine
! CONVERT: define void @s1_
! CONVERT: %[[LOAD_IGNORE:[0-9]+]] = load i64, ptr %"s1_str$len
! CONVERT: %[[LOAD:[0-9]+]] = load i64, ptr %"s1_str$len
! CONVERT: %[[TRUNC:[0-9]+]] = trunc i64 %[[LOAD]] to i32
! CONVERT: store i32 %[[TRUNC]], ptr %s1_test_len
! CONVERT_I8: define void @s1_
! CONVERT_I8: %[[LOAD_IGNORE:[0-9]+]] = load i64, ptr %"s1_str$len
! CONVERT_I8: %[[LOAD:[0-9]+]] = load i64, ptr %"s1_str$len
! CONVERT_I8: store i64 %[[LOAD]], ptr %s1_test_len

subroutine s2()
  character(len=100) :: s2_str
  integer :: s2_test_len
  s2_test_len = len(s2_str)
  print *, s2_test_len
end subroutine
! NO_CONVERT: define void @s2_
! NO_CONVERT: store i32 100, ptr %s2_test_len
! NO_CONVERT_I8: define void @s2_
! NO_CONVERT_I8: store i64 100, ptr %s2_test_len
