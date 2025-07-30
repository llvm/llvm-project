! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

! CHECK-LABEL: unroll_and_jam_dir
subroutine unroll_and_jam_dir
  integer :: a(10)
  !dir$ unroll_and_jam 4
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  ! CHECK-NOT: !llvm.loop
  ! CHECK:   br label {{.*}}, !llvm.loop ![[ANNOTATION:.*]]
  do i=1,10
     a(i)=i
  end do
end subroutine unroll_and_jam_dir

! CHECK-LABEL: unroll_and_jam_dir_0
subroutine unroll_and_jam_dir_0
  integer :: a(10)
  !dir$ unroll_and_jam 0
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  ! CHECK-NOT: !llvm.loop
  ! CHECK:   br label {{.*}}, !llvm.loop ![[ANNOTATION_DISABLE:.*]]
  do i=1,10
  a(i)=i
  end do
end subroutine unroll_and_jam_dir_0

! CHECK-LABEL: unroll_and_jam_dir_1
subroutine unroll_and_jam_dir_1
  integer :: a(10)
  !dir$ unroll_and_jam 1 
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  ! CHECK-NOT: !llvm.loop
  ! CHECK:   br label {{.*}}, !llvm.loop ![[ANNOTATION_DISABLE]]
  do i=1,10
  a(i)=i
  end do
end subroutine unroll_and_jam_dir_1

! CHECK-LABEL: nounroll_and_jam_dir
subroutine nounroll_and_jam_dir
  integer :: a(10)
  !dir$ nounroll_and_jam
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  ! CHECK-NOT: !llvm.loop
  ! CHECK:   br label {{.*}}, !llvm.loop ![[ANNOTATION_DISABLE]]
  do i=1,10
  a(i)=i
  end do
end subroutine nounroll_and_jam_dir

! CHECK-LABEL: unroll_and_jam_dir_no_factor
subroutine unroll_and_jam_dir_no_factor
  integer :: a(10)
  !dir$ unroll_and_jam
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  ! CHECK-NOT: !llvm.loop
  ! CHECK:   br label {{.*}}, !llvm.loop ![[ANNOTATION_NO_FACTOR:.*]]
  do i=1,10
  a(i)=i
  end do
end subroutine unroll_and_jam_dir_no_factor

! CHECK: ![[ANNOTATION]] = distinct !{![[ANNOTATION]], ![[UNROLL_AND_JAM:.*]], ![[UNROLL_AND_JAM_COUNT:.*]]}
! CHECK: ![[UNROLL_AND_JAM]] = !{!"llvm.loop.unroll_and_jam.enable"}
! CHECK: ![[UNROLL_AND_JAM_COUNT]] = !{!"llvm.loop.unroll_and_jam.count", i32 4}
! CHECK: ![[ANNOTATION_DISABLE]] = distinct !{![[ANNOTATION_DISABLE]], ![[UNROLL_AND_JAM2:.*]]}
! CHECK: ![[UNROLL_AND_JAM2]] = !{!"llvm.loop.unroll_and_jam.disable"}
! CHECK: ![[ANNOTATION_NO_FACTOR]] = distinct !{![[ANNOTATION_NO_FACTOR]], ![[UNROLL_AND_JAM]]}
