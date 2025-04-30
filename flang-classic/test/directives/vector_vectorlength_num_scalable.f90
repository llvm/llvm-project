! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Check that the vector vectorlength(scalable) directive generates the correct metadata.
! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK-00,CHECK-ALL

! Check that "-Hx,59,2" disables vector directive.
! RUN: %flang -Hx,59,2 -S -emit-llvm %s -o - | FileCheck %s --implicit-check-not "llvm.loop.vectorize"

subroutine func1(a, b, m)
! CHECK-ALL: define void @func1
  integer :: i, m, a(m), b(m)
  !dir$ vector vectorlength(2,scalable)
  do i = 1, m
    b(i) = a(i) + 1
  end do
! CHECK-00:      [[LOOP:L.LB[0-9]_[0-9]+]]:{{[' ',\t]+}}; preds = %[[LOOP]], %L.LB
! CHECK-00:      br i1 {{.*}}, label %[[LOOP]], {{.*}} !llvm.loop [[LOOP_LATCH_MD:![0-9]+]]
end subroutine func1

! CHECK-00-NOT:  !"llvm.loop.vectorize.width"
! CHECK-00:      [[VE_MD:![0-9]+]] = !{!"llvm.loop.vectorize.enable", i1 true}
! CHECK-00:      [[VS_MD:![0-9]+]] = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
! CHECK-00:      [[VW_MD:![0-9]+]] = !{!"llvm.loop.vectorize.width", i32 2}
! CHECK-00:      [[LOOP_LATCH_MD]] = distinct !{
! CHECK-00-SAME: [[VE_MD]]
! CHECK-00-SAME: [[VS_MD]]
! CHECK-00-SAME: [[VW_MD]]
! CHECK-00-SAME: }
