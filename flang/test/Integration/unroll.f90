! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

! CHECK-LABEL: unroll_dir
subroutine unroll_dir
  integer :: a(10)
  !dir$ unroll
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[UNROLL_ENABLE_FULL_ANNO:.*]]
  do i=1,10
  a(i)=i
  end do
end subroutine unroll_dir

! CHECK-LABEL: unroll_dir_0
subroutine unroll_dir_0
  integer :: a(10)
  !dir$ unroll 0
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[UNROLL_DISABLE_ANNO:.*]]
  do i=1,10
  a(i)=i
  end do
end subroutine unroll_dir_0

! CHECK-LABEL: unroll_dir_1
subroutine unroll_dir_1
  integer :: a(10)
  !dir$ unroll 1
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[UNROLL_DISABLE_ANNO]]
  do i=1,10
  a(i)=i
  end do
end subroutine unroll_dir_1

! CHECK-LABEL: unroll_dir_2
subroutine unroll_dir_2
  integer :: a(10)
  !dir$ unroll 2
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[UNROLL_ENABLE_COUNT_2:.*]]
  do i=1,10
  a(i)=i
  end do
end subroutine unroll_dir_2

! CHECK: ![[UNROLL_ENABLE_FULL_ANNO]] = distinct !{![[UNROLL_ENABLE_FULL_ANNO]], ![[UNROLL_ENABLE:.*]], ![[UNROLL_FULL:.*]]}
! CHECK: ![[UNROLL_ENABLE:.*]] = !{!"llvm.loop.unroll.enable"}
! CHECK: ![[UNROLL_FULL:.*]] = !{!"llvm.loop.unroll.full"}
! CHECK: ![[UNROLL_DISABLE_ANNO]] = distinct !{![[UNROLL_DISABLE_ANNO]], ![[UNROLL_DISABLE:.*]]}
! CHECK: ![[UNROLL_DISABLE]] = !{!"llvm.loop.unroll.disable"}
! CHECK: ![[UNROLL_ENABLE_COUNT_2]] = distinct !{![[UNROLL_ENABLE_COUNT_2]], ![[UNROLL_ENABLE]], ![[UNROLL_COUNT_2:.*]]}
! CHECK: ![[UNROLL_COUNT_2]] = !{!"llvm.loop.unroll.count", i32 2}
