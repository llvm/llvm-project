! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! The alignment requested with !DIR$ ALIGN is recorded on the object, and a
! request that is ignored leaves nothing behind.

module m
  !dir$ align(64) :: a
  !dir$ align(32) :: al, cm, eq
  real :: a(10), cm, eq, eq2
  real, allocatable :: al(:)
  common /blk/ cm
  equivalence (eq, eq2)
end module

! CHECK: a, PUBLIC {{.*}}alignment: 64
! CHECK-NOT: alignment:
