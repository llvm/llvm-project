! Test lowering of Cray pointees in OpenACC constructs. The descriptor of the
! pointee is associated with its Cray pointer on the host, before the data
! clause or the compute construct. Compute constructs do not associate it, so
! that device code does not read the host Cray pointer.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPimplicit
subroutine implicit(pool)
  real :: pool(*)
  real :: z(4); pointer (ip_z, z)
  integer :: i
  ip_z = loc(pool(1))
  !$acc parallel loop
  do i = 1, 4
    z(i) = i
  end do
end subroutine
! CHECK:       fir.call @_FortranAPointerAssociateScalar
! CHECK-NEXT:  acc.parallel
! CHECK-NOT:   _FortranAPointerAssociateScalar
! CHECK:       acc.yield

! CHECK-LABEL: func.func @_QPexplicit
subroutine explicit(pool)
  real :: pool(*)
  real :: z(4); pointer (ip_z, z)
  integer :: i
  ip_z = loc(pool(1))
  !$acc data present(z)
  !$acc parallel loop
  do i = 1, 4
    z(i) = i
  end do
  !$acc end data
end subroutine
! CHECK:       fir.call @_FortranAPointerAssociateScalar
! CHECK-NEXT:  %[[PRESENT:.*]] = acc.present varPtr({{.*}}) name("z")
! CHECK:       acc.data dataOperands(%[[PRESENT]]
! CHECK:       fir.call @_FortranAPointerAssociateScalar
! CHECK-NEXT:  acc.parallel
! CHECK-NOT:   _FortranAPointerAssociateScalar
! CHECK:       acc.yield

! Outside of compute constructs, references keep their association.
! CHECK-LABEL: func.func @_QPhost
subroutine host(pool)
  real :: pool(*)
  real :: z(4); pointer (ip_z, z)
  ip_z = loc(pool(1))
  z(1) = 1.0
end subroutine
! CHECK:       fir.call @_FortranAPointerAssociateScalar
