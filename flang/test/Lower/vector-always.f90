! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK: #loop_vectorize = #llvm.loop_vectorize<disable = false>
! CHECK: #loop_vectorize1 = #llvm.loop_vectorize<disable = true>
! CHECK: #loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize>
! CHECK: #loop_annotation1 = #llvm.loop_annotation<vectorize = #loop_vectorize1>

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
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
  end do
end subroutine intermediate_directive


! CHECK-LABEL: no_vector
subroutine no_vector
  integer :: a(10)
  !dir$ novector
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation1}
  do i=1,10
     a(i)=i
  end do
end subroutine no_vector
