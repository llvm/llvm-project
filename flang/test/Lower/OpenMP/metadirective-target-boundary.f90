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
! statement keeps its source position before an inner PARALLEL.
! CHECK-LABEL: func.func @_QPdeclarative_target_parallel()
! CHECK: omp.target
! CHECK: omp.parallel
! CHECK: omp.taskyield
! CHECK: return
subroutine declarative_target_parallel()
  !$omp metadirective when(implementation={vendor(llvm)}: target)
  block
    !$omp parallel
      !$omp metadirective when(construct={target, parallel}: taskyield) &
      !$omp& default(nothing)
    !$omp end parallel
  end block
end subroutine

! A BLOCK associated with a selected TARGET is mapped like a direct TARGET body.
! The TARGET region's binding of x does not outlive it.
! CHECK-LABEL: func.func @_QPselected_target_block(
! CHECK: %[[X:[0-9]+]]{{(:2)?}} = {{(hlfir|fir)}}.declare %{{.*}} uniq_name("_QFselected_target_blockEx")
! CHECK: %[[MAP:.*]] = omp.map.info {{.*}}name("x")
! CHECK: omp.target {{.*}}map_entries(%[[MAP]] -> %[[ARG:.*]] : !fir.ref<i32>) {
! CHECK: {{(hlfir|fir)}}.declare %[[ARG]] uniq_name("_QFselected_target_blockEx")
! CHECK: omp.terminator
! CHECK: fir.load %[[X]]{{(#0)?}} :
subroutine selected_target_block(x)
  integer :: x
  !$omp metadirective when(implementation={vendor(llvm)}: target)
  block
    x = x + 1
  end block
  x = x + 2
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

! Compiler directives between the METADIRECTIVE and the BLOCK are not BLOCK
! references.
! CHECK-LABEL: func.func @_QMm_directivesPselected_target_block_directives(
! CHECK-NOT: omp.map.info {{.*}}name("x")
! CHECK-NOT: omp.map.info {{.*}}name("a")
! CHECK: %[[Y:.*]] = omp.map.info {{.*}}name("y")
! CHECK-NOT: omp.map.info
! CHECK: omp.target {{.*}}map_entries(%[[Y]] -> %{{[^ ]*}} : !fir.ref<i32>) {
! CHECK: omp.terminator
module m_directives
contains
  subroutine selected_target_block_directives(x, a, y)
    integer :: x(10), a(10), y
    !$omp metadirective when(implementation={vendor(llvm)}: target)
    !dir$ ignore_tkr (k) x
    !dir$ assume_aligned a:64
    block
      y = y + 1
    end block
    y = y + 2
  end subroutine
end module
