! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

! Simple character assignment tests
! CHECK-LABEL: assign1
subroutine assign1(lhs, rhs)
  character(*, 1) :: lhs, rhs
  lhs = rhs
  ! Unboxing
  ! CHECK-DAG:[[lhs:%[0-9]+]]:2 = fir.unboxchar %arg0
  ! CHECK-DAG:[[rhs:%[0-9]+]]:2 = fir.unboxchar %arg1

  ! Compute minimum length
  ! CHECK-DAG:%[[cmp_len:[0-9]+]] = cmpi "slt", [[lhs]]#1, [[rhs]]#1
  ! CHECK-DAG:[[min_len:%[0-9]+]] = select %[[cmp_len]], [[lhs]]#1, [[rhs]]#1

  ! Allocate temp in case rhs and lhs may overlap
  ! CHECK: [[tmp:%[0-9]+]] = fir.alloca !fir.char<1>, [[min_len]]

  ! Copy of rhs into temp
  ! CHECK: fir.loop [[i:%[[:alnum:]_]+]]
    ! CHECK-DAG: [[rhs_addr:%[0-9]+]] = fir.coordinate_of [[rhs]]#0, [[i]]
    ! CHECK-DAG: [[tmp_addr:%[0-9]+]] = fir.coordinate_of [[tmp]], [[i]]
    ! CHECK-DAG: [[rhs_elt:%[0-9]+]] = fir.load [[rhs_addr]]
    ! CHECK: fir.store [[rhs_elt]] to [[tmp_addr]]
  ! CHECK: }

  ! Copy of temp into lhs
  ! CHECK: fir.loop [[i:%[[:alnum:]]+]]
    ! CHECK-DAG: [[tmp_addr:%[0-9]+]] = fir.coordinate_of [[tmp]], [[i]]
    ! CHECK-DAG: [[lhs_addr:%[0-9]+]] = fir.coordinate_of [[lhs]]#0, [[i]]
    ! CHECK-DAG: [[tmp_elt:%[0-9]+]] = fir.load [[tmp_addr]]
    ! CHECK: fir.store [[tmp_elt]] to [[lhs_addr]]
  ! CHECK: }

  ! Padding
  ! CHECK: [[c32:%[[:alnum:]_]+]] = constant 32 : i8
  ! CHECK: [[blank:%[0-9]+]] = fir.convert [[c32]] : (i8) -> !fir.char<1>
  ! CHECK: fir.loop [[i:%[[:alnum:]_]+]]
    ! CHECK-DAG: [[lhs_addr:%[0-9]+]] = fir.coordinate_of [[lhs]]#0, [[i]]
    ! CHECK: fir.store [[blank]] to [[lhs_addr]]
  ! CHECK: }
end subroutine

! Test substring assignment
! CHECK-LABEL: assign_substring1
subroutine assign_substring1(str, rhs, lb, ub)
  character(*, 1) :: rhs, str
  integer(8) :: lb, ub
  str(lb:ub) = rhs
  ! CHECK-DAG: [[lb:%[0-9]+]] = fir.load %arg2
  ! CHECK-DAG: [[ub:%[0-9]+]] = fir.load %arg3
  ! CHECK-DAG: [[str:%[0-9]+]]:2 = fir.unboxchar %arg0

  ! Compute substring offset
  ! CHECK-DAG: [[lbi:%[0-9]+]] = fir.convert [[lb]] : (i64) -> index
  ! CHECK-DAG: [[c1:%[[:alnum:]_]+]] = constant 1
  ! CHECK-DAG: [[offset:%[0-9]+]] = subi [[lbi]], [[c1]]
  ! CHECK-DAG: [[lhs_addr:%[0-9]+]] = fir.coordinate_of [[str]]#0, [[offset]]


  ! Compute substring length
  ! CHECK-DAG: [[diff:%[0-9]+]] = subi [[ub]], [[lb]]
  ! CHECK-DAG: [[c1:%[[:alnum:]_]+]] = constant 1
  ! CHECK-DAG: [[pre_lhs_len:%[0-9]+]] = addi [[diff]], [[c1]]
  ! CHECK-DAG: [[c0:%[[:alnum:]_]+]] = constant 0
  ! CHECK-DAG: [[cmp_len:%[0-9]+]] = cmpi "slt", [[pre_lhs_len]], [[c0]]
  ! CHECK-DAG: [[lhs_len:%[0-9]+]] = select [[cmp_len]], [[c0]], [[pre_lhs_len]]

  ! CHECK: [[lhs_box:%[0-9]+]] = fir.emboxchar [[lhs_addr]], [[lhs_len]]

  ! The rest of the assignment is just as the one above, only test that the
  ! substring box is the one used
  ! ...
  ! CHECK: [[lhs:%[0-9]+]]:2 = fir.unboxchar [[lhs_box]]
  ! ...
  ! CHECK: fir.coordinate_of [[lhs]]#0, {{.*}}
  ! ...
end subroutine

! CHECK-LABEL: assign_constant
! CHECK: (%[[ARG:.*]]:{{.*}})
subroutine assign_constant(lhs)
  character(*, 1) :: lhs
  ! CHECK-DAG: %[[lhs:.*]]:2 = fir.unboxchar %[[ARG]] :
  ! CHECK-DAG: %[[tmp:.*]] = fir.address_of(@{{.*}}) :
  lhs = "Hello World"
  ! CHECK: fir.loop %[[i:.*]] = %{{.*}} to %{{.*}} {
    ! CHECK-DAG: %[[tmp_addr:.*]] = fir.coordinate_of %[[tmp]], %[[i]]
    ! CHECK-DAG: %[[tmp_elt:.*]] = fir.load %[[tmp_addr]]
    ! CHECK-DAG: %[[lhs_addr:.*]] = fir.coordinate_of %[[lhs]]#0, %[[i]]
    ! CHECK: fir.store %[[tmp_elt]] to %[[lhs_addr]]
  ! CHECK: }

  ! Padding
  ! CHECK-DAG: %[[c32:.*]] = constant 32 : i8
  ! CHECK-DAG: %[[blank:.*]] = fir.convert %[[c32]] : (i8) -> !fir.char<1>
  ! CHECK: fir.loop %[[j:.*]] = %{{.*}} to %{{.*}} {
    ! CHECK: %[[jhs_addr:.*]] = fir.coordinate_of %[[lhs]]#0, %[[j]]
    ! CHECK: fir.store %[[blank]] to %[[jhs_addr]]
  ! CHECK: }
end subroutine

! CHECK-LABEL: fir.global @_QQ48656C6C6F20576F726C64
! CHECK: %[[lit:.*]] = fir.string_lit "Hello World"(11) : !fir.char<1>
! CHECK: fir.has_value %[[lit]] : !fir.array<11x!fir.char<1>>
! CHECK: }
