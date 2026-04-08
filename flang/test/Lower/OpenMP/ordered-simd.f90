! This test checks lowering of SIMD constructs with ordered regions.
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

! Test that ordered regions inside SIMD have par_level_simd attribute
subroutine ordered_simd(n)
  integer :: n, a(n), b(n), c(n), i

! CHECK-LABEL: func @_QPordered_simd
! CHECK:         omp.simd linear({{.*}}) private({{.*}}) {
! CHECK:           omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
! CHECK:             omp.ordered.region par_level_simd {
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.yield
! CHECK:           }
! CHECK:         } {linear_var_types = [i32]}

  !$omp simd
  do i = 1, n
    a(i) = b(i) * 10
    !$omp ordered simd
    print *, a(i)
    !$omp end ordered
    c(i) = a(i) * 2
  end do
  !$omp end simd

end subroutine

! Test that ordered regions inside DO SIMD have par_level_simd attribute
subroutine ws_ordered_simd(n)
  integer :: n, a(n), b(n), c(n), i

! CHECK-LABEL: func @_QPws_ordered_simd
! CHECK:         omp.wsloop ordered(0) {
! CHECK:           omp.simd linear({{.*}}) private({{.*}}) {
! CHECK:             omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
! CHECK:               omp.ordered.region par_level_simd {
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.yield
! CHECK:             }
! CHECK:           } {linear_var_types = [i32], omp.composite}
! CHECK:         } {omp.composite}

  !$omp do simd ordered
  do i = 1, n
    a(i) = b(i) * 10
    !$omp ordered simd
    print *, a(i)
    !$omp end ordered
    c(i) = a(i) * 2
  end do
  !$omp end do simd

end subroutine
