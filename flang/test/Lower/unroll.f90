! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK: #loop_unroll = #llvm.loop_unroll<disable = false, full = true>
! CHECK: #loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll>

! CHECK-LABEL: unroll_dir
subroutine unroll_dir
  integer :: a(10)
  !dir$ unroll
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
  end do
end subroutine unroll_dir


! CHECK-LABEL: intermediate_directive
subroutine intermediate_directive
  integer :: a(10)
  !dir$ unroll
  !dir$ unknown
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
  end do
end subroutine intermediate_directive

