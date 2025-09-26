! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK: #loop_vectorize = #llvm.loop_vectorize<ivdepEnable = true>
! CHECK: #loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize>

! CHECK-LABEL: @_QPivdep
subroutine ivdep
  integer :: a(10)
  !dir$ ivdep 
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
  end do
end subroutine ivdep


! CHECK-LABEL: @_QPintermediate_directive
subroutine intermediate_directive
  integer :: a(10)
  !dir$ ivdep 
  !dir$ unknown
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
  end do
end subroutine intermediate_directive
