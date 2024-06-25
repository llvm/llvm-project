!RUN: bbc -fopenmp -emit-hlfir -o - %s | FileCheck %s

!https://github.com/llvm/llvm-project/issues/91205

!CHECK: omp.parallel if(%{{[0-9]+}} : i1) {
!CHECK:   %[[THP1:[0-9]+]] = omp.threadprivate %{{[0-9]+}}#1
!CHECK:   %[[DCL1:[0-9]+]]:2 = hlfir.declare %[[THP1]] {uniq_name = "_QFcopyin_scalar_arrayEx1"}
!CHECK:   %[[LD1:[0-9]+]] = fir.load %{{[0-9]+}}#0
!CHECK:   hlfir.assign %[[LD1]] to %[[DCL1]]#0 temporary_lhs
!CHECK:   %[[THP2:[0-9]+]] = omp.threadprivate %{{[0-9]+}}#1
!CHECK:   %[[SHP2:[0-9]+]] = fir.shape %c{{[0-9]+}}
!CHECK:   %[[DCL2:[0-9]+]]:2 = hlfir.declare %[[THP2]](%[[SHP2]]) {uniq_name = "_QFcopyin_scalar_arrayEx2"}
!CHECK:   hlfir.assign %{{[0-9]+}}#0 to %[[DCL2]]#0 temporary_lhs
!CHECK:   omp.barrier
!CHECK:   fir.call @_QPsub1(%[[DCL1]]#1, %[[DCL2]]#1)
!CHECK:   omp.terminator
!CHECK: }

subroutine copyin_scalar_array()
  integer(kind=4), save :: x1
  integer(kind=8), save :: x2(10)
  !$omp threadprivate(x1, x2)

  ! Have x1 appear before x2 in the AST node for the `parallel` construct,
  ! but at the same time have them in a different order in `copyin`.
  !$omp parallel if (x1 .eq. x2(1)) copyin(x2, x1)
    call sub1(x1, x2)
  !$omp end parallel

end

