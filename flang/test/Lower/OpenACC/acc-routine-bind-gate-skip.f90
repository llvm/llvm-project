! A decorated acc routine bind(...) whose procedure is never lowered gets no
! bind-target declaration and no acc.routine.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine s_gate_skip()
  !$acc routine(unused_ext) seq bind(unused_ext_dev)
  external :: unused_ext
end subroutine

! CHECK-LABEL: func.func @_QPs_gate_skip
! CHECK-NOT: @_QPunused_ext_dev
! CHECK-NOT: @_QPunused_ext
! CHECK-NOT: acc.routine
