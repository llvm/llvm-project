! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! Check that the UNROLL directive generates correct LLVM IR metadata at -O0.
! Each subroutine should have distinct metadata, particularly subroutines with
! different unroll factors specified by the user.
!
! RUN: %flang -O0 -S -emit-llvm %s -o - \
! RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-O0

! Check that LLVM unrolls the first loop fully at -O1, unrolls the other two
! loops the correct number of times, and disables further unrolling on them.
!
! RUN: %flang -O1 -S -emit-llvm %s -o - \
! RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-O1

! Check that "-Hx,59,2" disables both kinds of UNROLL directives.
!
! RUN: %flang -Hx,59,2 -S -emit-llvm %s -o - \
! RUN: | FileCheck %s --check-prefixes=CHECK,CHECK-DISABLED

subroutine func1(a, b)
  ! CHECK-LABEL: define void @func1_
  integer :: m = 10
  integer :: i, a(m), b(m)

  !dir$ unroll
  do i = 1, m
    b(i) = a(i) + 1
  end do
  ! CHECK-O0:           [[BB1:L.LB[0-9]_[0-9]+]]:{{[ \t]+}}; preds = %[[BB1]],
  ! CHECK-O0:           br i1 {{.*}}, label %[[BB1]]
  ! CHECK-O0-SAME:      !llvm.loop [[MD_LOOP1:![0-9]+]]
  ! CHECK-O1-COUNT-10:  store i32
  ! CHECK-O1-NOT:       store i32
  ! CHECK-O1-NOT:       br i1 {{.*}}, label %{{L.LB[0-9]_[0-9]+}}
  ! CHECK-O1-NOT:       !llvm.loop !{{[0-9]+}}
  ! CHECK-O1:           ret void
  ! CHECK-DISABLED:     [[BB1:L.LB[0-9]_[0-9]+]]:{{[ \t]+}}; preds = %[[BB1]],
  ! CHECK-DISABLED:     br i1 {{.*}}, label %[[BB1]]
end subroutine func1

subroutine func2(m, a, b)
  ! CHECK-LABEL: define void @func2_
  integer :: i, m, a(m), b(m)

  !dir$ unroll(4)
  do i = 1, m
    b(i) = a(i) + 1
  end do
  ! CHECK:              [[BB2:L.LB[0-9]_[0-9]+]]:{{[ \t]+}}; preds = %[[BB2]],
  ! CHECK-O1-COUNT-4:   store i32
  ! CHECK-O1-NOT:       store i32
  ! CHECK:              br i1 {{.*}}, label %[[BB2]]
  ! CHECK-O0-SAME:      !llvm.loop [[MD_LOOP2:![0-9]+]]
  ! CHECK-O1-SAME:      !llvm.loop [[MD_LOOP2:![0-9]+]]
end subroutine func2

subroutine func3(m, a, b)
  ! CHECK-LABEL: define void @func3_
  integer :: i, m, a(m), b(m)

  ! Use an odd factor to make sure it's picked up.
  !dir$ unroll = 7
  do i = 1, m
    b(i) = a(i) + 1
  end do
  ! CHECK:              [[BB3:L.LB[0-9]_[0-9]+]]:{{[ \t]+}}; preds = %[[BB3]],
  ! CHECK-O1-COUNT-7:   store i32
  ! CHECK-O1-NOT:       store i32
  ! CHECK:              br i1 {{.*}}, label %[[BB3]]
  ! CHECK-O0-SAME:      !llvm.loop [[MD_LOOP3:![0-9]+]]
  ! CHECK-O1-SAME:      !llvm.loop [[MD_LOOP3:![0-9]+]]
end subroutine func3

! CHECK-O0-DAG: [[MD_ENABLE:![0-9]+]] = !{!"llvm.loop.unroll.enable"}
! CHECK-O0-DAG: [[MD_LOOP1]] = distinct !{[[MD_LOOP1]], {{.*}}, {{.*}}, [[MD_ENABLE]]}
! CHECK-O0-DAG: [[MD_COUNT1:![0-9]+]] = !{!"llvm.loop.unroll.count", i32 4}
! CHECK-O0-DAG: [[MD_LOOP2]] = distinct !{[[MD_LOOP2]], {{.*}}, {{.*}}, [[MD_COUNT1]]}
! CHECK-O0-DAG: [[MD_COUNT2:![0-9]+]] = !{!"llvm.loop.unroll.count", i32 7}
! CHECK-O0-DAG: [[MD_LOOP3]] = distinct !{[[MD_LOOP3]], {{.*}}, {{.*}}, [[MD_COUNT2]]}

! CHECK-O1-NOT: !"llvm.loop.unroll.enable"
! CHECK-O1:     [[MD_LOOP2]] = distinct !{[[MD_LOOP2]], {{.*}}, {{.*}}, [[MD_DISABLE:![0-9]+]]}
! CHECK-O1:     [[MD_DISABLE]] = !{!"llvm.loop.unroll.disable"}
! CHECK-O1:     [[MD_LOOP3]] = distinct !{[[MD_LOOP3]], {{.*}}, {{.*}}, [[MD_DISABLE]]}

! CHECK-DISABLED-NOT: !"llvm.loop.unroll.enable"
! CHECK-DISABLED-NOT: !"llvm.loop.unroll.count"
! CHECK-DISABLED-NOT: !"llvm.loop.unroll.disable"
