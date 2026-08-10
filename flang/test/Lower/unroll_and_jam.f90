! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK: #loop_unroll_and_jam = #llvm.loop_unroll_and_jam<disable = false>
! CHECK: #loop_unroll_and_jam1 = #llvm.loop_unroll_and_jam<disable = false, count = 2 : i64>
! CHECK: #loop_unroll_and_jam2 = #llvm.loop_unroll_and_jam<disable = true>
! CHECK: #loop_annotation = #llvm.loop_annotation<unrollAndJam = #loop_unroll_and_jam>
! CHECK: #loop_annotation1 = #llvm.loop_annotation<unrollAndJam = #loop_unroll_and_jam1>
! CHECK: #loop_annotation2 = #llvm.loop_annotation<unrollAndJam = #loop_unroll_and_jam2>

! CHECK-LABEL: unroll_and_jam_dir
subroutine unroll_and_jam_dir
  integer :: a(10)
  !dir$ unroll_and_jam
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
  end do

  !dir$ unroll_and_jam 2
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation1}
  do i=1,10
     a(i)=i
  end do
end subroutine unroll_and_jam_dir


! CHECK-LABEL: intermediate_directive
subroutine intermediate_directive
  integer :: a(10)
  !dir$ unroll_and_jam
  !dir$ unknown
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation}
  do i=1,10
     a(i)=i
  end do
end subroutine intermediate_directive


! CHECK-LABEL: nounroll_and_jam_dir
subroutine nounroll_and_jam_dir
  integer :: a(10)
  !dir$ nounroll_and_jam
  !CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #loop_annotation2}
  do i=1,10
     a(i)=i
  end do
end subroutine nounroll_and_jam_dir
