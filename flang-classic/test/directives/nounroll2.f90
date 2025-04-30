! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Check that Flang at -O0 does not unroll the loop, and does not generate any
! loop unrolling metadata.
!
! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-O0
!
! CHECK-O0:     [[LOOP:L.LB[0-9]_[0-9]+]]:{{[ \t]+}}; preds = %[[LOOP]], %L.LB
! CHECK-O0:     store float
! CHECK-O0-NOT: store float
! CHECK-O0:     br i1 {{.*}}, label %[[LOOP]], label %L.LB
! CHECK-O0-NOT: !"llvm.loop.unroll.disable"

! Check that LLVM vectorizes the loop automatically at -O2.
!
! RUN: %flang -O2 -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-O2
!
! CHECK-O2: vector.body:{{[ \t]+}}; preds = %vector.body, %L.
! CHECK-O2: br i1 {{.*}}, label %vector.body, !llvm.loop

program tz
  integer :: i
  real ::acc(10000)
  do i = 1, 10000
    acc(i) = i * 2.0
  end do
  print *, acc(1000)
end program
