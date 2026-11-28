! Check that the IAND reduction identity for INTEGER(16) has all 128 bits set
! rather than only the low 64 bits.

! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: fir.declare_reduction @iand_reduction_i128 : i128 init {
! CHECK:         %[[ONES:.*]] = arith.constant -1 : i128
! CHECK:         fir.yield(%[[ONES]] : i128)

subroutine reduce_iand_i128(a, n, r)
  integer :: n, i
  integer(16) :: a(n), r
  do concurrent (i = 1:n) reduce(iand: r)
    r = iand(r, a(i))
  end do
end subroutine
