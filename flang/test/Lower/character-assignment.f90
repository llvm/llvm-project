! RUN: bbc --use-desc-for-alloc=false %s -o - -emit-fir | FileCheck %s

! Simple character assignment tests
! CHECK-LABEL: _QPassign1
subroutine assign1(lhs, rhs)
  character(*, 1) :: lhs, rhs
  ! CHECK: %[[lhs:.*]]:2 = fir.unboxchar %arg0
  ! CHECK: %[[rhs:.*]]:2 = fir.unboxchar %arg1
  lhs = rhs
  ! Compute minimum length
  ! CHECK: %[[cmp_len:[0-9]+]] = arith.cmpi slt, %[[lhs:.*]]#1, %[[rhs:.*]]#1
  ! CHECK-NEXT: %[[min_len:[0-9]+]] = arith.select %[[cmp_len]], %[[lhs]]#1, %[[rhs]]#1

  ! Copy of rhs into lhs
  ! CHECK: %[[count:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
  ! CHECK-DAG: %[[bug:.*]] = fir.convert %[[lhs]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[src:.*]] = fir.convert %[[rhs]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%{{.*}}, %[[src]], %[[count]], %false) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()

  ! Padding
  ! CHECK-DAG: %[[blank:.*]] = fir.insert_value %{{.*}}, %c32{{.*}}, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK: fir.do_loop %[[ij:.*]] =
    ! CHECK-DAG: %[[lhs_cast:.*]] = fir.convert %[[lhs]]#0
    ! CHECK: %[[lhs_addr:.*]] = fir.coordinate_of %[[lhs_cast]], %[[ij]]
    ! CHECK: fir.store %[[blank]] to %[[lhs_addr]]
  ! CHECK-NEXT: }
end subroutine

! Test substring assignment
! CHECK-LABEL: _QPassign_substring1
subroutine assign_substring1(str, rhs, lb, ub)
  character(*, 1) :: rhs, str
  integer(8) :: lb, ub
  str(lb:ub) = rhs
  ! CHECK-DAG: %[[lb:.*]] = fir.load %arg2
  ! CHECK-DAG: %[[ub:.*]] = fir.load %arg3
  ! CHECK-DAG: %[[str:.*]]:2 = fir.unboxchar %arg0

  ! Compute substring offset
  ! CHECK-DAG: %[[lbi:.*]] = fir.convert %[[lb]] : (i64) -> index
  ! CHECK-DAG: %[[c1:.*]] = arith.constant 1
  ! CHECK-DAG: %[[offset:.*]] = arith.subi %[[lbi]], %[[c1]]
  ! CHECK-DAG: %[[str_cast:.*]] = fir.convert %[[str]]#0
  ! CHECK-DAG: %[[str_addr:.*]] = fir.coordinate_of %[[str_cast]], %[[offset]]
  ! CHECK-DAG: %[[lhs_addr:.*]] = fir.convert %[[str_addr]]

  ! Compute substring length
  ! CHECK-DAG: %[[ubi:.*]] = fir.convert %[[ub]] : (i64) -> index
  ! CHECK-DAG: %[[diff:.*]] = arith.subi %[[ubi]], %[[lbi]]
  ! CHECK-DAG: %[[pre_lhs_len:.*]] = arith.addi %[[diff]], %[[c1]]
  ! CHECK-DAG: %[[c0:.*]] = arith.constant 0
  ! CHECK-DAG: %[[cmp_len:.*]] = arith.cmpi slt, %[[pre_lhs_len]], %[[c0]]

  ! CHECK-DAG: %[[lhs_len:.*]] = arith.select %[[cmp_len]], %[[c0]], %[[pre_lhs_len]]

  ! The rest of the assignment is just as the one above, only test that the
  ! substring is the one used as lhs.
  ! ...
  ! CHECK: fir.do_loop %arg4 =
  ! CHECK: %[[lhs_addr3:.*]] = fir.convert %[[lhs_addr]]
  ! CHECK: fir.coordinate_of %[[lhs_addr3]], %arg4
  ! ...
end subroutine

! CHECK-LABEL: _QPassign_constant
! CHECK-SAME: (%[[ARG:.*]]:
subroutine assign_constant(lhs)
  character(*, 1) :: lhs
  ! CHECK: %[[lhs:.*]]:2 = fir.unboxchar %arg0
  ! CHECK: %[[cst:.*]] = fir.address_of(@{{.*}}) :
  lhs = "Hello World"
  ! CHECK-DAG: %[[dst:.*]] = fir.convert %[[lhs]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[src:.*]] = fir.convert %[[cst]] : (!fir.ref<!fir.char<1,11>>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[count:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[dst]], %[[src]], %[[count]], %false) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()

  ! Padding
  ! CHECK-DAG: %[[blank:.*]] = fir.insert_value %{{.*}}, %c32{{.*}}, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK: fir.do_loop %[[j:.*]] = %{{.*}} to %{{.*}} {
    ! CHECK-DAG: %[[jhs_cast:.*]] = fir.convert %[[lhs]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
    ! CHECK: %[[jhs_addr:.*]] = fir.coordinate_of %[[jhs_cast]], %[[j]]
    ! CHECK: fir.store %[[blank]] to %[[jhs_addr]]
  ! CHECK: }
end subroutine

  ! CHECK: func @_QPassign_zero_size_array
  subroutine assign_zero_size_array(n)
    ! CHECK:   %[[VAL_0:.*]] = fir.alloca !fir.heap<!fir.array<?x!fir.char<1,?>>> {uniq_name = "_QFassign_zero_size_arrayEa.addr"}
    character(n), allocatable :: a(:)
    ! CHECK:   fir.store %{{.*}} to %[[VAL_0]] : !fir.ref<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
    ! CHECK:   %{{.*}} = fir.load %[[VAL_0]] : !fir.ref<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
    ! CHECK:   %[[VAL_1:.*]] = arith.cmpi ne, %{{.*}}, %c0{{.*}} : i64
    ! CHECK:   %[[VAL_2:.*]]:2 = fir.if %[[VAL_1]] -> (i1, !fir.heap<!fir.array<?x!fir.char<1,?>>>) {
    ! CHECK:     %{{.*}} = fir.if %{{.*}} -> (!fir.heap<!fir.array<?x!fir.char<1,?>>>) {
    ! CHECK:   %{{.*}} = fir.do_loop %{{.*}} = %c0{{.*}} to %{{.*}} step %c1{{.*}} unordered iter_args(%{{.*}} = %{{.*}}) -> (!fir.array<?x!fir.char<1,?>>) {
    ! CHECK:     fir.do_loop %[[ARG_0:.*]] = %{{.*}} to {{.*}} step %c1{{.*}} {
    ! CHECK:       %{{.*}} = fir.coordinate_of %{{.*}}, %[[ARG_0]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
    ! CHECK:   fir.if %[[VAL_2]]#0 {
    ! CHECK:     fir.if %[[VAL_1]] {
    ! CHECK:     fir.store %[[VAL_2]]#1 to %[[VAL_0]] : !fir.ref<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
    a = [character(n)::]
    ! CHECK:   return
  end subroutine

! CHECK-LABEL: fir.global internal @_QQcl.48656C6C6F20576F726C64
! CHECK: %[[lit:.*]] = fir.string_lit "Hello World"(11) : !fir.char<1,11>
! CHECK: fir.has_value %[[lit]] : !fir.char<1,11>
! CHECK: }
