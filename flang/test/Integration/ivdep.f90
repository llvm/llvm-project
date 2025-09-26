! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

! CHECK-LABEL: ivdep 
subroutine ivdep
  integer :: a(10)
  !dir$ ivdep
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  ! CHECK-NOT: !llvm.loop
  ! CHECK:   br label {{.*}}, !llvm.loop ![[ANNOTATION:.*]]
  do i=1,10
     a(i)=i
  end do
end subroutine ivdep

! CHECK: ![[ANNOTATION]] = distinct !{![[ANNOTATION]], ![[IVDEP:.*]]}
! CHECK: ![[IVDEP]] = !{!"llvm.loop.vectorize.ivdep.enable", i1 true}
