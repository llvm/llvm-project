! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK: #access_group = #llvm.access_group<id = distinct[0]<>>
! CHECK: #access_group1 = #llvm.access_group<id = distinct[1]<>>
! CHECK: #loop_vectorize = #llvm.loop_vectorize<disable = false>
! CHECK: #loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize, parallelAccesses = #access_group>
! CHECK: #loop_annotation1 = #llvm.loop_annotation<vectorize = #loop_vectorize, parallelAccesses = #access_group1>

! CHECK-LABEL: vector_always
subroutine vector_always
  integer :: a(10)
  !dir$ vector always
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
  end do
end subroutine vector_always


! CHECK-LABEL: intermediate_directive
subroutine intermediate_directive
  integer :: a(10)
  !dir$ vector always
  !dir$ unknown
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation1}
  do i=1,10
     a(i)=i
  end do
end subroutine intermediate_directive
