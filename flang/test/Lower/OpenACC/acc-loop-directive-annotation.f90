! Test that !dir$ loop directives are applied as loopAnnotation on acc.loop.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine acc_loop_unroll(a, n)
  real :: a(n)
  integer :: i, n
  !dir$ unroll
  !$acc loop
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_loop_unroll
! CHECK: acc.loop {{.*}} {
! CHECK: } attributes {{{.*}}loopAnnotation = #llvm.loop_annotation<unroll = <disable = false, full = true>>}
