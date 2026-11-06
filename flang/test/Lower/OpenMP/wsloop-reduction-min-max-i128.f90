! Check that the MIN and MAX reduction identities for INTEGER(16) are the
! full 128-bit limits rather than values truncated to 64 bits.

! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: omp.declare_reduction @max_i128 : i128 init {
! CHECK:         %[[MIN:.*]] = arith.constant -170141183460469231731687303715884105728 : i128
! CHECK:         omp.yield(%[[MIN]] : i128)

! CHECK-LABEL: omp.declare_reduction @min_i128 : i128 init {
! CHECK:         %[[MAX:.*]] = arith.constant 170141183460469231731687303715884105727 : i128
! CHECK:         omp.yield(%[[MAX]] : i128)

subroutine reduce_i128(a, n, r, q)
  integer :: n, i
  integer(16) :: a(n), r, q
  !$omp parallel do reduction(min:r) reduction(max:q)
  do i = 1, n
    r = min(r, a(i))
    q = max(q, a(i))
  end do
end subroutine
