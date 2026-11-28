! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s


! CHECK-LABEL: func.func @_QPone(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK: %[[VAL_2:.*]] = fir.alloca i32
! CHECK: %[[VAL_8:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFoneEa")
! CHECK: %[[VAL_9:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} uniq_name("_QFoneEs")
! CHECK: mif.alloc_coarray %[[VAL_8]]#0 {{.*}} stat %[[VAL_2]] {{.*}} uniq_name("_QFoneEa")
! CHECK: %[[VAL_17:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK: fir.store %[[VAL_17]] to %[[VAL_9]]#0 : !fir.ref<i32>
subroutine one(s)
  integer, allocatable :: a(:)[:]
  integer :: s
  allocate(a(4)[*], stat=s)
end subroutine

! CHECK-LABEL: func.func @_QPtwo(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "s"}) {
! CHECK: %[[VAL_2:.*]] = fir.alloca i32
! CHECK: %[[VAL_5:.*]] = fir.alloca i32
! CHECK: %[[VAL_11:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFtwoEa")
! CHECK: %[[VAL_16:.*]]:2 = hlfir.declare {{.*}} uniq_name("_QFtwoEb")
! CHECK: %[[VAL_17:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} uniq_name("_QFtwoEs")
! CHECK: mif.alloc_coarray %[[VAL_11]]#0 {{.*}} stat %[[VAL_5]] {{.*}} uniq_name("_QFtwoEa")
! CHECK: %[[VAL_25:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK: fir.store %[[VAL_25]] to %[[VAL_17]]#0 : !fir.ref<i32>
! CHECK: %[[C0_I32_2:.*]] = arith.constant 0 : i32
! CHECK: %[[VAL_26:.*]] = arith.cmpi eq, %[[VAL_25]], %[[C0_I32_2]] : i32
! CHECK: fir.if %[[VAL_26]] {
! CHECK: mif.alloc_coarray %[[VAL_16]]#0 {{.*}} stat %[[VAL_2]] {{.*}} uniq_name("_QFtwoEb")
! CHECK: %[[VAL_41:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK: fir.store %[[VAL_41]] to %[[VAL_17]]#0 : !fir.ref<i32>
! CHECK-NEXT: }
subroutine two(s)
  integer, allocatable :: a(:)[:], b(:)[:]
  integer :: s
  allocate(a(4)[*], b(4)[*], stat=s)
end subroutine

! CHECK-LABEL: func.func @_QPalloc_kind8(
! CHECK:   %[[VAL_2:.*]] = fir.alloca i32
! CHECK:   %[[VAL_9:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 uniq_name("_QFalloc_kind8Es")
! CHECK:   mif.alloc_coarray {{.*}} stat %[[VAL_2]] errmsg
! CHECK-NEXT:   %[[VAL_17:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK-NEXT:   %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK-NEXT:   fir.store %[[VAL_18]] to %[[VAL_9]]#0 : !fir.ref<i64>
subroutine alloc_kind8(s)
  integer(8) :: s
  integer, allocatable :: a(:)[:]
  allocate(a(4)[*], stat=s)
end subroutine

! CHECK-LABEL: func.func @_QPalloc_kind2(
! CHECK:   %[[VAL_2:.*]] = fir.alloca i32
! CHECK:   %[[VAL_9:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} arg 1 uniq_name("_QFalloc_kind2Es")
! CHECK:   mif.alloc_coarray {{.*}} stat %[[VAL_2]] errmsg
! CHECK-NEXT:   %[[VAL_17:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK-NEXT:   %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i16
! CHECK-NEXT:   fir.store %[[VAL_18]] to %[[VAL_9]]#0 : !fir.ref<i16>
subroutine alloc_kind2(s)
  integer(2) :: s
  integer, allocatable :: a(:)[:]
  allocate(a(4)[*], stat=s)
end subroutine
