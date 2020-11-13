! RUN: bbc %s -o - -emit-fir | FileCheck %s

! Simple character assignment tests
! CHECK-LABEL: assign1
subroutine assign1(lhs, rhs)
  character(*, 1) :: lhs, rhs
  ! CHECK-DAG: %[[lhs:.*]]:2 = fir.unboxchar %arg0
  ! CHECK-DAG: %[[rhs:.*]]:2 = fir.unboxchar %arg1
  lhs = rhs
  ! Compute minimum length
  ! CHECK: %[[cmp_len:[0-9]+]] = cmpi "slt", %[[lhs:.*]]#1, %[[rhs:.*]]#1
  ! CHECK-NEXT: %[[min_len:[0-9]+]] = select %[[cmp_len]], %[[lhs]]#1, %[[rhs]]#1

  ! Allocate temp in case rhs and lhs may overlap
  ! CHECK: %[[tmp:.*]] = fir.alloca !fir.char<1,?>, %[[min_len]]

  ! Copy of rhs into temp
  ! CHECK: fir.do_loop %[[i:.*]] =
    ! CHECK-DAG: %[[rhs_cast:.*]] = fir.convert %[[rhs]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
    ! CHECK-DAG: %[[rhs_addr:.*]] = fir.coordinate_of %[[rhs_cast]], %[[i]]
    ! CHECK-DAG: %[[rhs_elt:.*]] = fir.load %[[rhs_addr]]
    ! CHECK-DAG: %[[tmp_cast:.*]] = fir.convert %[[tmp]]
    ! CHECK-DAG: %[[tmp_addr:.*]] = fir.coordinate_of %[[tmp_cast]], %[[i]]
    ! CHECK: fir.store %[[rhs_elt]] to %[[tmp_addr]]
  ! CHECK-NEXT: }

  ! Copy of temp into lhs
  ! CHECK: fir.do_loop %[[ii:.*]] =
    ! CHECK-DAG: %[[tmp_cast:.*]] = fir.convert %[[tmp]]
    ! CHECK-DAG: %[[tmp_addr:.*]] = fir.coordinate_of %[[tmp_cast]], %[[ii]]
    ! CHECK-DAG: %[[tmp_elt:.*]] = fir.load %[[tmp_addr]]
    ! CHECK-DAG: %[[lhs_cast:.*]] = fir.convert %[[lhs]]#0
    ! CHECK-DAG: %[[lhs_addr:.*]] = fir.coordinate_of %[[lhs_cast]], %[[ii]]
    ! CHECK: fir.store %[[tmp_elt]] to %[[lhs_addr]]
  ! CHECK-NEXT: }

  ! Padding
  ! CHECK-DAG: %[[blank:.*]] = fir.insert_value %{{.*}}, %c32{{.*}}, %c0{{.*}} : (!fir.char<1>, i8, index) -> !fir.char<1>
  ! CHECK: fir.do_loop %[[ij:.*]] =
    ! CHECK-DAG: %[[lhs_cast:.*]] = fir.convert %[[lhs]]#0
    ! CHECK: %[[lhs_addr:.*]] = fir.coordinate_of %[[lhs_cast]], %[[ij]]
    ! CHECK: fir.store %[[blank]] to %[[lhs_addr]]
  ! CHECK-NEXT: }
end subroutine

! Test substring assignment
! CHECK-LABEL: assign_substring1
subroutine assign_substring1(str, rhs, lb, ub)
  character(*, 1) :: rhs, str
  integer(8) :: lb, ub
  str(lb:ub) = rhs
  ! CHECK-DAG: %[[lb:.*]] = fir.load %arg2
  ! CHECK-DAG: %[[ub:.*]] = fir.load %arg3
  ! CHECK-DAG: %[[str:.*]]:2 = fir.unboxchar %arg0

  ! Compute substring offset
  ! CHECK-DAG: %[[lbi:.*]] = fir.convert %[[lb]] : (i64) -> index
  ! CHECK-DAG: %[[c1:.*]] = constant 1
  ! CHECK-DAG: %[[offset:.*]] = subi %[[lbi]], %[[c1]]
  ! CHECK-DAG: %[[str_cast:.*]] = fir.convert %[[str]]#0
  ! CHECK-DAG: %[[str_addr:.*]] = fir.coordinate_of %[[str_cast]], %[[offset]]
  ! CHECK-DAG: %[[lhs_addr:.*]] = fir.convert %[[str_addr]]

  ! Compute substring length
  ! CHECK-DAG: %[[ubi:.*]] = fir.convert %[[ub]] : (i64) -> index
  ! CHECK-DAG: %[[diff:.*]] = subi %[[ubi]], %[[lbi]]
  ! CHECK-DAG: %[[pre_lhs_len:.*]] = addi %[[diff]], %[[c1]]
  ! CHECK-DAG: %[[c0:.*]] = constant 0
  ! CHECK-DAG: %[[cmp_len:.*]] = cmpi "slt", %[[pre_lhs_len]], %[[c0]]

  ! CHECK-DAG: %[[lhs_len:.*]] = select %[[cmp_len]], %[[c0]], %[[pre_lhs_len]]

  ! The rest of the assignment is just as the one above, only test that the
  ! substring is the one used as lhs.
  ! ...
  ! CHECK: %[[lhs_addr3:.*]] = fir.convert %[[lhs_addr]]
  ! CHECK-NEXT: fir.coordinate_of %[[lhs_addr3]], %arg4
  ! ...
end subroutine

! CHECK-LABEL: assign_constant
! CHECK: (%[[ARG:.*]]:{{.*}})
subroutine assign_constant(lhs)
  character(*, 1) :: lhs
  ! CHECK: %[[lhs:.*]]:2 = fir.unboxchar %arg0
  ! CHECK: %[[cst:.*]] = fir.address_of(@{{.*}}) :
  ! CHECK: %[[tmp:.*]] = fir.alloca !fir.char<1,?>, %{{.*}}
  lhs = "Hello World"
  ! CHECK: fir.do_loop %[[i:.*]] = %{{.*}} to %{{.*}} {
    ! CHECK: %[[cst2:.*]] = fir.convert %[[cst]]
    ! CHECK-DAG: %[[cst_addr:.*]] = fir.coordinate_of %[[cst2]], %[[i]]
    ! CHECK-DAG: %[[cst_elt:.*]] = fir.load %[[cst_addr]]
    ! CHECK: %[[lhs_addr:.*]] = fir.coordinate_of %[[tmp:.*]], %[[i]]
    ! CHECK: fir.store %[[cst_elt]] to %[[lhs_addr]]
  ! CHECK: }

  ! Padding
  ! CHECK-DAG: %[[blank:.*]] = fir.insert_value %{{.*}}, %c32{{.*}}, %c0{{.*}} : (!fir.char<1>, i8, index) -> !fir.char<1>
  ! CHECK: fir.do_loop %[[j:.*]] = %{{.*}} to %{{.*}} {
    ! CHECK-DAG: %[[jhs_cast:.*]] = fir.convert %[[lhs]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
    ! CHECK: %[[jhs_addr:.*]] = fir.coordinate_of %[[jhs_cast]], %[[j]]
    ! CHECK: fir.store %[[blank]] to %[[jhs_addr]]
  ! CHECK: }
end subroutine

! CHECK-LABEL: fir.global linkonce @_QQcl.48656C6C6F20576F726C64
! CHECK: %[[lit:.*]] = fir.string_lit "Hello World"(11) : !fir.char<1,11>
! CHECK: fir.has_value %[[lit]] : !fir.char<1,11>
! CHECK: }
