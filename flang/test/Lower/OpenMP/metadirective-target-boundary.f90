! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=51 -emit-hlfir %s -o - | \
! RUN:   FileCheck %s

! TARGET hides the outer PARALLEL, so the SIMD replacement is not lowered.
! CHECK-LABEL: func.func @_QPactual_target(
! CHECK: omp.parallel
! CHECK: omp.target
! CHECK-NOT: omp.simd
! CHECK: return
subroutine actual_target(n, a)
  integer :: n, i, a(n)
  !$omp parallel
    !$omp target
      !$omp metadirective &
      !$omp& when(construct={parallel}: simd) default(nothing)
      do i = 1, n
        a(i) = i
      end do
    !$omp end target
  !$omp end parallel
end subroutine

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

! A selected TARGET from a metadirective that precedes every executable
! statement keeps its source position before an inner TEAMS.
! CHECK-LABEL: func.func @_QPdeclarative_target_teams()
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: omp.taskyield
! CHECK: return
subroutine declarative_target_teams()
  !$omp metadirective when(implementation={vendor(llvm)}: target)
  block
    !$omp teams
      !$omp metadirective when(construct={target, teams}: taskyield) &
      !$omp& default(nothing)
    !$omp end teams
  end block
end subroutine

! A BLOCK associated with a selected TARGET is mapped like a direct TARGET body.
! CHECK-LABEL: func.func @_QPselected_target_block(
! CHECK: %[[MAP:.*]] = omp.map.info {{.*}}name("x")
! CHECK: omp.target {{.*}}map_entries(%[[MAP]] -> %[[ARG:.*]] : !fir.ref<i32>) {
! CHECK: {{(hlfir|fir)}}.declare %[[ARG]] uniq_name("_QFselected_target_blockEx")
! CHECK: omp.terminator
subroutine selected_target_block(x)
  integer :: x
  !$omp metadirective when(implementation={vendor(llvm)}: target)
  block
    x = x + 1
  end block
end subroutine

! A named constant with a dynamic substring in the BLOCK is mapped.
! CHECK-LABEL: func.func @_QPselected_target_block_substring(
! CHECK: %[[P:.*]] = omp.map.info {{.*}}name("p")
! CHECK: omp.target {{.*}}map_entries({{.*}}%[[P]] -> %[[ARG:[^ ,]*]]
! CHECK: {{(hlfir|fir)}}.declare %[[ARG]] {{.*}}uniq_name("_QFselected_target_block_substringECp")
! CHECK: omp.terminator
subroutine selected_target_block_substring(n, c)
  integer :: n
  character(1) :: c
  character(4), parameter :: p = "abcd"
  !$omp metadirective when(implementation={vendor(llvm)}: target)
  block
    c = p(n:n)
  end block
end subroutine
