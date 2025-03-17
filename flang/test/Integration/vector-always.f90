! RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

! CHECK-LABEL: vector_always
subroutine vector_always
  integer :: a(10)
  !dir$ vector always
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  ! CHECK-NOT: !llvm.loop
  ! CHECK:   br label {{.*}}, !llvm.loop ![[ANNOTATION:.*]]
  do i=1,10
     a(i)=i
  end do
end subroutine vector_always

! CHECK: ![[ANNOTATION]] = distinct !{![[ANNOTATION]], ![[VECTORIZE:.*]]}
! CHECK: ![[VECTORIZE]] = !{!"llvm.loop.vectorize.enable", i1 true}
