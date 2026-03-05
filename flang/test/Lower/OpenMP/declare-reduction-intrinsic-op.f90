! RUN: %flang_fc1 -emit-mlir -fopenmp %s -o - | FileCheck %s

program test
  type t
     integer :: x
  end type t
  !$omp declare reduction(+:t: omp_out%x = omp_out%x + omp_in%x) initializer(omp_priv = t(0))
  type(t) :: a
  a = t(0)
  !$omp parallel reduction(+:a)
  a%x = a%x + 1
  !$omp end parallel
end program test

! CHECK: omp.declare_reduction @add_reduction_byref_rec__QFTt : !fir.ref<!fir.type<_QFTt{x:i32}>>
! CHECK-SAME: attributes {byref_element_type = !fir.type<_QFTt{x:i32}>}
! CHECK: alloc {
! CHECK:   omp.yield
! CHECK: } init {
! CHECK:   omp.yield
! CHECK: } combiner {
! CHECK:   omp.yield
! CHECK: }
