! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

! CHECK-LABEL: unroll_dir
subroutine unroll_dir
  integer :: a(10)
  !dir$ unroll 
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[ANNOTATION:.*]]
  do i=1,10
     a(i)=i
  end do
end subroutine unroll_dir

! CHECK: ![[ANNOTATION]] = distinct !{![[ANNOTATION]], ![[UNROLL:.*]], ![[UNROLL_FULL:.*]]}
! CHECK: ![[UNROLL]] = !{!"llvm.loop.unroll.enable"}
! CHECK: ![[UNROLL_FULL]] = !{!"llvm.loop.unroll.full"}

