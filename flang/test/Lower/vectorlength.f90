! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK: #[[FIXED:.*]] = #llvm.loop_vectorize<disable = false, scalableEnable = false>
! CHECK: #[[SCALABLE:.*]] = #llvm.loop_vectorize<disable = false, scalableEnable = true>
! CHECK: #[[WIDTH2:.*]] = #llvm.loop_vectorize<disable = false, width = 2 : i64>
! CHECK: #[[FIXED_WIDTH2:.*]] = #llvm.loop_vectorize<disable = false, scalableEnable = false, width = 2 : i64>
! CHECK: #[[SCALABLE_WIDTH2:.*]] = #llvm.loop_vectorize<disable = false, scalableEnable = true, width = 2 : i64>
! CHECK: #[[FIXED_TAG:.*]] = #llvm.loop_annotation<vectorize = #[[FIXED]]>
! CHECK: #[[SCALABLE_TAG:.*]] = #llvm.loop_annotation<vectorize = #[[SCALABLE]]>
! CHECK: #[[WIDTH2_TAG:.*]]  = #llvm.loop_annotation<vectorize = #[[WIDTH2]]>
! CHECK: #[[FIXED_WIDTH2_TAG:.*]] = #llvm.loop_annotation<vectorize = #[[FIXED_WIDTH2]]>
! CHECK: #[[SCALABLE_WIDTH2_TAG:.*]] = #llvm.loop_annotation<vectorize = #[[SCALABLE_WIDTH2]]>

! CHECK-LABEL: func.func @_QPfixed(
subroutine fixed(a, b, m)
  integer :: i, m, a(m), b(m)

  !dir$ vector vectorlength(fixed)
  ! CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[FIXED_TAG]]}
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPscalable(
subroutine scalable(a, b, m)
  integer :: i, m, a(m), b(m)

  !dir$ vector vectorlength(scalable)
  ! CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[SCALABLE_TAG]]}
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPlen2(
subroutine len2(a, b, m)
  integer :: i, m, a(m), b(m)

  !dir$ vector vectorlength(2)
  ! CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[WIDTH2_TAG]]}
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPlen2fixed(
subroutine len2fixed(a, b, m)
  integer :: i, m, a(m), b(m)

  !dir$ vector vectorlength(2,fixed)
  ! CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[FIXED_WIDTH2_TAG]]}
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPlen2scalable(
subroutine len2scalable(a, b, m)
  integer :: i, m, a(m), b(m)

  !dir$ vector vectorlength(2,scalable)
  ! CHECK: fir.do_loop {{.*}} attributes {loopAnnotation = #[[SCALABLE_WIDTH2_TAG]]}
  do i = 1, m
    b(i) = a(i) + 1
  end do
end subroutine
