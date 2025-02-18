! Test delayed privatization for derived types with allocatable components.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

subroutine firstprivate_alloc_comp
  type t1
    integer, allocatable :: c(:)
  end type
  type(t1) :: x
  !$omp parallel firstprivate(x)
    print *, allocated(x%c)
  !$omp end parallel
end

  call firstprivate_alloc_comp()
end
! CHECK-LABEL:   omp.private {type = firstprivate} @_QFfirstprivate_alloc_compEx_firstprivate_ref_rec__QFfirstprivate_alloc_compTt1 : !fir.ref<!fir.type<_QFfirstprivate_alloc_compTt1{c:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> alloc {
! CHECK:     fir.call @_FortranAInitialize(
! CHECK:   } copy {
! ...
