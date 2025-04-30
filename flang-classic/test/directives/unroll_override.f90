! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! When multiple UNROLL directives are specified for the same loop, check
! that the last one overrides previous ones.
!
! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s

subroutine func1(m, a, b)
  ! CHECK-LABEL: define void @func1_
  integer :: i, m, a(m), b(m)

  !dir$ nounroll
  !dir$ unroll
  !dir$ unroll = 10
  do i = 1, m
    b(i) = a(i) + 1
  end do
  ! CHECK:       [[BB1:L.LB[0-9]_[0-9]+]]:{{[ \t]+}}; preds = %[[BB1]],
  ! CHECK:       br i1 {{.*}}, label %[[BB1]]
  ! CHECK-SAME:  !llvm.loop [[MD_LOOP1:![0-9]+]]
end subroutine func1

subroutine func2(m, a, b)
  ! CHECK-LABEL: define void @func2_
  integer :: i, m, a(m), b(m)

  !dir$ unroll = 10
  !dir$ nounroll
  !dir$ unroll
  do i = 1, m
    b(i) = a(i) + 1
  end do
  ! CHECK:       [[BB2:L.LB[0-9]_[0-9]+]]:{{[ \t]+}}; preds = %[[BB2]],
  ! CHECK:       br i1 {{.*}}, label %[[BB2]]
  ! CHECK-SAME:  !llvm.loop [[MD_LOOP2:![0-9]+]]
end subroutine func2

subroutine func3(m, a, b)
  ! CHECK-LABEL: define void @func3_
  integer :: i, m, a(m), b(m)

  !dir$ unroll
  !dir$ unroll = 10
  !dir$ nounroll
  do i = 1, m
    b(i) = a(i) + 1
  end do
  ! CHECK:       [[BB3:L.LB[0-9]_[0-9]+]]:{{[ \t]+}}; preds = %[[BB3]],
  ! CHECK:       br i1 {{.*}}, label %[[BB3]]
  ! CHECK-SAME:  !llvm.loop [[MD_LOOP3:![0-9]+]]
end subroutine func3

! Check that metadata are correct.
!
! CHECK: [[MD_COUNT:![0-9]+]] = !{!"llvm.loop.unroll.count", i32 10}
! CHECK: [[MD_LOOP1]] = distinct !{[[MD_LOOP1]], {{.*}}, {{.*}}, [[MD_COUNT]]}
! CHECK: [[MD_ENABLE:![0-9]+]] = !{!"llvm.loop.unroll.enable"}
! CHECK: [[MD_LOOP2]] = distinct !{[[MD_LOOP2]], {{.*}}, {{.*}}, [[MD_ENABLE]]}
! CHECK: [[MD_DISABLE:![0-9]+]] = !{!"llvm.loop.unroll.disable"}
! CHECK: [[MD_LOOP3]] = distinct !{[[MD_LOOP3]], {{.*}}, {{.*}}, [[MD_DISABLE]]}
