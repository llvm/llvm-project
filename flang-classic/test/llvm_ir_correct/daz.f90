!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang -target x86_64-unknown-unknown -Mdaz %s -S -emit-llvm -o - | FileCheck %s
! REQUIRES: x86_64-host
! CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__daz, ptr null }]

program daz
end program

! CHECK: declare void @__daz()
