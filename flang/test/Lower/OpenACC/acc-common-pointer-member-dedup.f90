! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! A POINTER member of a COMMON block that is listed together with the whole
! block is contained in the block and is dropped, as for any other member. The
! member is then firstprivatized with the bytes of the block, which copies the
! pointer association, instead of through an operand of its own. The region
! derives the member from the block operand.

subroutine block_then_pointer_member()
  integer :: k
  integer, pointer :: p
  common /blk/ k, p
  !$acc parallel firstprivate(/blk/) firstprivate(p)
    k = 1
    p = 2
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPblock_then_pointer_member
! CHECK-NOT: acc.firstprivate {{.*}} name("p")
! CHECK: %[[BLOCK:.*]] = acc.firstprivate varPtr({{.*}}) recipe({{.*}}) name("blk") -> !fir.ref<!fir.array<32xi8>>
! CHECK-NOT: acc.firstprivate
! CHECK: acc.parallel firstprivate(%[[BLOCK]] : !fir.ref<!fir.array<32xi8>>) {
! CHECK: hlfir.declare {{.*}} storage(%[[BLOCK]][0])
! CHECK: hlfir.declare {{.*}} storage(%[[BLOCK]][8]) {{.*}}fortran_attrs<pointer>
! CHECK: acc.yield

subroutine pointer_member_then_block()
  integer :: k
  integer, pointer :: p
  common /blk/ k, p
  !$acc parallel firstprivate(p) firstprivate(/blk/)
    k = 1
    p = 2
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPpointer_member_then_block
! CHECK-NOT: acc.firstprivate {{.*}} name("p")
! CHECK: %[[BLOCK:.*]] = acc.firstprivate varPtr({{.*}}) recipe({{.*}}) name("blk") -> !fir.ref<!fir.array<32xi8>>
! CHECK-NOT: acc.firstprivate
! CHECK: acc.parallel firstprivate(%[[BLOCK]] : !fir.ref<!fir.array<32xi8>>) {
! CHECK: hlfir.declare {{.*}} storage(%[[BLOCK]][0])
! CHECK: hlfir.declare {{.*}} storage(%[[BLOCK]][8]) {{.*}}fortran_attrs<pointer>
! CHECK: acc.yield
