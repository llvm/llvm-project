! RUN: not %flang_fc1 -fsyntax-only -J%S/Inputs %s 2>&1 | FileCheck %s

! Inputs/mbaddim.mod was produced from the module below, then hand-edited to
! corrupt the serialized rank-1-bound (ROBE) inquiries and recompute the
! module-file checksum so it is read, not rejected as corrupt:
!   b: `ubound(a,dim=2,kind=8)` -> `ubound(a,dim=9,kind=8)`
!   d: `ubound(a,dim=2,kind=8)` -> `ubound(a,dim=2147483648_8,kind=8)` (2^31)
! `a` has rank 2, so both DIMs are out of range (the 2^31 value also exercises
! that a DIM past INT32_MAX is reported, not silently wrapped).  Consuming the
! module must re-analyze the serialized bound expressions and report the
! out-of-range DIM gracefully rather than crash, proving a ROBE inquiry
! round-tripped through a module file inherits the ordinary DIM-range check.

#if 0
module mbaddim
contains
  subroutine s(n, m, c, a, b, d)
    integer, intent(in) :: n(2), m(2)
    logical, intent(in) :: c(2)
    real :: a(merge(n, m, c))
    real :: b(ubound(a,2))
    real :: d(ubound(a,2))
    b(1) = 1.0
    a(1,1) = b(1)
  end subroutine
end module
#endif

! CHECK-DAG: error: DIM=9 dimension is out of range for rank-2 array
! CHECK-DAG: error: DIM=2147483648 dimension is out of range for rank-2 array
program test
  use mbaddim
end program
