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

! CHECK-LABEL: no_vector
subroutine no_vector
  integer :: a(10)
  !dir$ novector
  ! CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}
  ! CHECK-NOT: !llvm.loop
  ! CHECK:   br label {{.*}}, !llvm.loop ![[ANNOTATION2:.*]]
  do i=1,10
     a(i)=i
  end do
end subroutine no_vector

! CHECK: ![[ANNOTATION]] = distinct !{![[ANNOTATION]], ![[VECTORIZE:.*]]}
! CHECK: ![[VECTORIZE]] = !{!"llvm.loop.vectorize.enable", i1 true}
! CHECK: ![[ANNOTATION2]] = distinct !{![[ANNOTATION2]], ![[VECTORIZE2:.*]]}
! CHECK: ![[VECTORIZE2]] = !{!"llvm.loop.vectorize.enable", i1 false}
