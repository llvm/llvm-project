! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 -emit-hlfir %s -o - | FileCheck %s

! A selected TARGET contributes its own construct trait.
! CHECK-LABEL: func.func @_QPselected_target_present()
! CHECK: omp.parallel
! CHECK: omp.target
! CHECK-NOT: omp.taskyield
! CHECK: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine selected_target_present()
  !$omp parallel
    !$omp begin metadirective default(target)
      !$omp metadirective &
      !$omp& when(construct={target}: barrier) default(taskyield)
    !$omp end metadirective
  !$omp end parallel
end subroutine

! The same selected TARGET hides the enclosing PARALLEL trait.
! CHECK-LABEL: func.func @_QPselected_target_hides_parallel()
! CHECK: omp.parallel
! CHECK: omp.target
! CHECK-NOT: omp.barrier
! CHECK: omp.taskyield
! CHECK-NOT: omp.barrier
! CHECK: return
subroutine selected_target_hides_parallel()
  !$omp parallel
    !$omp begin metadirective default(target)
      !$omp metadirective &
      !$omp& when(construct={parallel}: barrier) default(taskyield)
    !$omp end metadirective
  !$omp end parallel
end subroutine

! The boundary includes TARGET itself and constructs nested inside it.
! CHECK-LABEL: func.func @_QPtarget_inner_parallel()
! CHECK: omp.parallel
! CHECK: omp.target
! CHECK: omp.parallel
! CHECK-NOT: omp.taskyield
! CHECK: omp.barrier
! CHECK-NOT: omp.taskyield
! CHECK: return
subroutine target_inner_parallel()
  !$omp parallel
    !$omp target
      !$omp parallel
        !$omp metadirective when(construct={target, parallel}: barrier) &
        !$omp& default(taskyield)
      !$omp end parallel
    !$omp end target
  !$omp end parallel
end subroutine
