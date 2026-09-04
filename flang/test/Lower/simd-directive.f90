! Test lowering of SIMD directive to ensure that correct cectorisation flags are
! applied to the do loops

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test1()
  integer :: i

  !DIR$ SIMD
  do i = 1, 10
  end do
end subroutine

! CHECK: #[[loop_vec:.*]] = #llvm.loop_vectorize<disable = false>
! CHECK: #[[loop_annotate:.*]] = #llvm.loop_annotation<vectorize = #[[loop_vec]]>

! CHECK :%[[.*]] = fir.do_loop %[[.*]] = %[[.*]] to %[[.*]] step %[[.*]] iter_args(%arg1 = %[[.*]]) -> (i32) attributes {loopAnnotation = #[[loop_annontate]]} {
