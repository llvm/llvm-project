! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

! CHECK-LABEL: unroll_and_jam_dir
subroutine unroll_and_jam_dir
  integer :: a(10)
  !dir$ unroll_and_jam 4
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[ANNOTATION:.*]]
  do i=1,10
     a(i)=i
  end do
end subroutine unroll_and_jam_dir

! CHECK: ![[ANNOTATION]] = distinct !{![[ANNOTATION]], ![[UNROLL_AND_JAM:.*]], ![[UNROLL_AND_JAM_COUNT:.*]]}
! CHECK: ![[UNROLL_AND_JAM]] = !{!"llvm.loop.unroll_and_jam.enable"}
! CHECK: ![[UNROLL_AND_JAM_COUNT]] = !{!"llvm.loop.unroll_and_jam.count", i32 4}
