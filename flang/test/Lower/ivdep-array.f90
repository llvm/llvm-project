! RUN: %flang_fc1 -emit-fir -O2 %s -o - | FileCheck %s

! CHECK: #[[ANNOTATION:.*]] = #llvm.loop_annotation<parallelAccesses = #[[GROUP:.*]]>
subroutine array_assignment_in_loop(a, b)
  real :: a(100,100), b(100,100)
  !dir$ ivdep
  ! CHECK: fir.do_loop
  ! CHECK-SAME: loopAnnotation = #[[ANNOTATION]]
  do i=1,100
    ! CHECK: fir.do_loop
      ! CHECK: fir.load
      ! CHECK-SAME: accessGroups = [#[[GROUP]]]
      ! CHECK: fir.store
      ! CHECK-SAME: accessGroups = [#[[GROUP]]]
    a(i, :) = b(i, :)
    ! CHECK: }
  ! CHECK: }
  ! CHECK: return
  end do
end subroutine
