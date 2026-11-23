! RUN: %flang_fc1 -emit-fir -O2 %s -o - | FileCheck %s

! CHECK: #[[ANNOTATION:.*]] = #llvm.loop_annotation<parallelAccesses = #[[GROUP:.*]]>
subroutine test_where(a, l)
  real :: a(100,100)
  logical :: l(100, 100)
  !dir$ ivdep
  ! CHECK: fir.do_loop
  ! CHECK-SAME: loopAnnotation = #[[ANNOTATION]]
  do i=1,100
    ! CHECK: fir.do_loop
    where (l(i, :)) a(i, :) = 3.0
    ! CHECK: fir.store
    ! CHECK-SAME: accessGroups = [#[[GROUP]]]
    ! CHECK: }
  end do
end subroutine
