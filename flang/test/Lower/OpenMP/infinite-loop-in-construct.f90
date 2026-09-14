! RUN: bbc -fopenmp -o - %s | FileCheck %s

! Check that this test can be lowered successfully.
! See https://github.com/llvm/llvm-project/issues/74348

! CHECK-LABEL:  func.func @_QPsb
! CHECK:          omp.parallel
! CHECK:            cf.cond_br %{{[0-9]+}}, ^bb1, ^bb4
! CHECK-NEXT:     ^bb1:  // pred: ^bb0
! CHECK:            cf.cond_br %{{[0-9]+}}, ^bb2, ^bb3
! CHECK-NEXT:     ^bb2:  // pred: ^bb1
! CHECK:            cf.br ^bb3
! CHECK-NEXT:     ^bb3:  // 2 preds: ^bb1, ^bb2
! CHECK:            cf.br ^bb4
! CHECK-NEXT:     ^bb4:  // 2 preds: ^bb0, ^bb3
! CHECK:            cf.br ^bb5
! CHECK-NEXT:     ^bb5:  // 2 preds: ^bb4, ^bb5
! CHECK:            cf.br ^bb5
! CHECK-NEXT:     }

subroutine sb(ninter, numnod)
  integer :: ninter, numnod
  integer, dimension(:), allocatable :: indx_nm

  !$omp parallel
  if (ninter>0) then
    allocate(indx_nm(numnod))
  endif
  220 continue
  goto 220
  !$omp end parallel
end subroutine
