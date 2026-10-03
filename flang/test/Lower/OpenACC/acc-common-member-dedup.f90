! RUN: bbc -fopenacc -emit-hlfir %s -o - 2>/dev/null | FileCheck %s

! Retain the whole COMMON operand and drop redundant explicit members.

subroutine private_block_first()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel private(/blk/) private(a)
    a = b + 1
    b = a + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPprivate_block_first
! CHECK-NOT: acc.private {{.*}} name("a")
! CHECK-NOT: acc.private {{.*}} name("b")
! CHECK: %[[BLOCK:.*]] = acc.private {{.*}} name("blk") -> !fir.ref<!fir.array<8xi8>>
! CHECK-NOT: acc.private {{.*}} name("a")
! CHECK-NOT: acc.private {{.*}} name("b")
! CHECK-NOT: acc.private {{.*}} name("blk")
! CHECK: acc.parallel{{.*}} private(%[[BLOCK]] : !fir.ref<!fir.array<8xi8>>) {
! CHECK-NOT: acc.private
! CHECK: hlfir.assign
! CHECK: hlfir.assign
! CHECK: acc.yield

subroutine private_member_first()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel private(a) private(/blk/)
    a = b + 1
    b = a + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPprivate_member_first
! CHECK-NOT: acc.private {{.*}} name("a")
! CHECK-NOT: acc.private {{.*}} name("b")
! CHECK: %[[BLOCK:.*]] = acc.private {{.*}} name("blk") -> !fir.ref<!fir.array<8xi8>>
! CHECK-NOT: acc.private {{.*}} name("a")
! CHECK-NOT: acc.private {{.*}} name("b")
! CHECK-NOT: acc.private {{.*}} name("blk")
! CHECK: acc.parallel{{.*}} private(%[[BLOCK]] : !fir.ref<!fir.array<8xi8>>) {
! CHECK-NOT: acc.private
! CHECK: hlfir.assign
! CHECK: hlfir.assign
! CHECK: acc.yield

subroutine private_members_first()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel private(a, b) private(/blk/)
    a = b + 1
    b = a + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPprivate_members_first
! CHECK-NOT: acc.private {{.*}} name("a")
! CHECK-NOT: acc.private {{.*}} name("b")
! CHECK: %[[BLOCK:.*]] = acc.private {{.*}} name("blk") -> !fir.ref<!fir.array<8xi8>>
! CHECK-NOT: acc.private {{.*}} name("a")
! CHECK-NOT: acc.private {{.*}} name("b")
! CHECK-NOT: acc.private {{.*}} name("blk")
! CHECK: acc.parallel{{.*}} private(%[[BLOCK]] : !fir.ref<!fir.array<8xi8>>) {
! CHECK-NOT: acc.private
! CHECK: hlfir.assign
! CHECK: hlfir.assign
! CHECK: acc.yield

subroutine firstprivate_block_first()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel firstprivate(/blk/) firstprivate(a)
    a = b + 1
    b = a + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPfirstprivate_block_first
! CHECK-NOT: acc.firstprivate {{.*}} name("a")
! CHECK-NOT: acc.firstprivate {{.*}} name("b")
! CHECK: %[[BLOCK:.*]] = acc.firstprivate {{.*}} name("blk") -> !fir.ref<!fir.array<8xi8>>
! CHECK-NOT: acc.firstprivate {{.*}} name("a")
! CHECK-NOT: acc.firstprivate {{.*}} name("b")
! CHECK-NOT: acc.firstprivate {{.*}} name("blk")
! CHECK: acc.parallel{{.*}} firstprivate(%[[BLOCK]] : !fir.ref<!fir.array<8xi8>>) {
! CHECK-NOT: acc.firstprivate
! CHECK: hlfir.assign
! CHECK: hlfir.assign
! CHECK: acc.yield

subroutine firstprivate_member_first()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel firstprivate(a) firstprivate(/blk/)
    a = b + 1
    b = a + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPfirstprivate_member_first
! CHECK-NOT: acc.firstprivate {{.*}} name("a")
! CHECK-NOT: acc.firstprivate {{.*}} name("b")
! CHECK: %[[BLOCK:.*]] = acc.firstprivate {{.*}} name("blk") -> !fir.ref<!fir.array<8xi8>>
! CHECK-NOT: acc.firstprivate {{.*}} name("a")
! CHECK-NOT: acc.firstprivate {{.*}} name("b")
! CHECK-NOT: acc.firstprivate {{.*}} name("blk")
! CHECK: acc.parallel{{.*}} firstprivate(%[[BLOCK]] : !fir.ref<!fir.array<8xi8>>) {
! CHECK-NOT: acc.firstprivate
! CHECK: hlfir.assign
! CHECK: hlfir.assign
! CHECK: acc.yield

subroutine repeated_whole_common()
  integer :: a, b
  common /blk/ a, b
  !$acc parallel private(/blk/) private(/blk/)
    a = b + 1
    b = a + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPrepeated_whole_common
! CHECK: %[[BLOCK:.*]] = acc.private {{.*}} name("blk") -> !fir.ref<!fir.array<8xi8>>
! CHECK-NOT: acc.private {{.*}} name("blk")
! CHECK: acc.parallel{{.*}} private(%[[BLOCK]] : !fir.ref<!fir.array<8xi8>>) {
! CHECK: hlfir.assign
! CHECK: hlfir.assign
! CHECK: acc.yield

subroutine common_array_container_last(i, j)
  integer :: a(10), b, i, j
  common /array_blk/ a, b
  !$acc parallel private(a(i), a(j)) private(/array_blk/)
    a(9) = b + 1
    b = a(9) + 1
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPcommon_array_container_last
! CHECK-NOT: acc.private {{.*}} name("a{{.*}}")
! CHECK: %[[BLOCK:.*]] = acc.private {{.*}} name("array_blk") -> !fir.ref<!fir.array<44xi8>>
! CHECK-NOT: acc.private
! CHECK: acc.parallel{{.*}} private(%[[BLOCK]] : !fir.ref<!fir.array<44xi8>>) {
! CHECK-NOT: acc.private
! CHECK: hlfir.assign
! CHECK: hlfir.assign
! CHECK: acc.yield
