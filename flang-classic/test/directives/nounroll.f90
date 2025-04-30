! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Check that the NOUNROLL directive generates the correct metadata.
!
! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
!
! CHECK:      [[LOOP:L.LB[0-9]_[0-9]+]]:{{[' ',\t]+}}; preds = %[[LOOP]], %L.LB
! CHECK:      store float
! CHECK-NOT:  store float
! CHECK:      br i1 {{.*}}, label %[[LOOP]], label %L.LB
! CHECK-SAME: !llvm.loop
! CHECK:      !"llvm.loop.unroll.disable"

! Check that "-Hx,59,2" disables the NOUNROLL directive.
!
! RUN: %flang -Hx,59,2 -S -emit-llvm %s -o - \
! RUN: | FileCheck %s --check-prefix=CHECK-NODIRECTIVE
!
! CHECK-NODIRECTIVE-NOT: !"llvm.loop.unroll.disable"

program tz
  integer :: i
  real :: acc(100)
  integer :: sz
  !dir$ nounroll
  do i = 1, sz
    acc(i) = i * 2.0
  end do
  print *, acc(100)
end program
