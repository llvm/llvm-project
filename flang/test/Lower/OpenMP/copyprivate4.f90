!RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!The second testcase from https://github.com/llvm/llvm-project/issues/141481

!Check that we don't crash on this.

!CHECK: omp.single copyprivate(%6#0 -> @_copy_class_ptr_rec__QFf01Tt : !fir.ref<!fir.class<!fir.ptr<!fir.type<_QFf01Tt>>>>) {
!CHECK:   omp.terminator
!CHECK: }

subroutine f01
  type t
  end type
  class(t), pointer :: tt

!$omp single copyprivate(tt)
!$omp end single
end
