! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!** Test checking attributes are set correctly
! REQUIRES: aarch64-registered-target
! REQUIRES: llvm-13

! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a %s -o - | FileCheck %s -check-prefix=ATTRS-NOSVE
! RUN: %flang -S -emit-llvm -target aarch64-linux-gnu -march=armv8-a+sve %s -o - | FileCheck %s -check-prefix=ATTRS-SVE
      program tz
       integer :: i
       integer :: acc(100)
       do i = 1, 100
            acc(i) = 5
       end do
       print *, acc(100)
      end program
! ATTRS-NOSVE: attributes{{.*}}"target-features"="+neon{{(,\+v8a)*}}"
! ATTRS-SVE: attributes{{.*}}"target-features"="+neon{{(,\+v8a)*}},+sve"
