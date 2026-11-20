! Check that the IAND reduction identity for INTEGER(16) has all 128 bits set
! rather than only the low 64 bits.

! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: omp.declare_reduction @iand_i128 : i128 init {
! CHECK:         %[[ONES:.*]] = arith.constant -1 : i128
! CHECK:         omp.yield(%[[ONES]] : i128)

subroutine reduce_iand_i128(a, n, r)
  integer :: n, i
  integer(16) :: a(n), r
  !$omp parallel do reduction(iand:r)
  do i = 1, n
    r = iand(r, a(i))
  end do
end subroutine
