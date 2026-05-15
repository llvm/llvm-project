! RUN: %flang_fc1 -emit-hlfir -fwrapv %s -o - | FileCheck %s

module cs
  type r
     integer n, d
  end type r

  type t2
     integer :: f1(5)
     type(r) :: f2
  end type t2

  type t3
     type(t2) :: f(3,3)
  end type t3

contains

  ! CHECK-LABEL: func.func @_QMcsPc1(
  ! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>{{.*}}) -> !fir.logical<1> {
  function c1(e, c)
    type(r), intent(in) :: e(:), c(:)
    logical*1 :: c1
    ! CHECK-DAG: %[[c1_ref:.*]] = fir.alloca !fir.logical<1>

    ! CHECK-DAG: %[[c_decl:.*]]:2 = hlfir.declare %[[arg1]] {{.*}}

    ! CHECK-DAG: %[[e_decl:.*]]:2 = hlfir.declare %[[arg0]] {{.*}}

    ! CHECK: %[[c_n:.*]] = hlfir.designate %[[c_decl]]#0{"n"}
    ! CHECK: %[[e_n:.*]] = hlfir.designate %[[e_decl]]#0{"n"}
    ! CHECK: %[[cmp_expr:.*]] = hlfir.elemental
    ! CHECK:   %[[c_addr:.*]] = hlfir.designate %[[c_n]]
    ! CHECK:   %[[e_addr:.*]] = hlfir.designate %[[e_n]]
    ! CHECK:   %[[c_val:.*]] = fir.load %[[c_addr]]
    ! CHECK:   %[[e_val:.*]] = fir.load %[[e_addr]]
    ! CHECK:   %[[cmp:.*]] = arith.cmpi eq, %[[c_val]], %[[e_val]]
    ! CHECK:   {{.*}} = fir.convert %[[cmp]] : (i1) -> !fir.logical<4>
    ! CHECK:   hlfir.yield_element
    ! CHECK: %[[all:.*]] = hlfir.all %[[cmp_expr]]
    ! CHECK: hlfir.assign %{{.*}} to %{{.*}} : !fir.logical<1>, !fir.ref<!fir.logical<1>>
    c1 = all(c%n == e%n)
  end function c1

  ! CHECK-LABEL: func.func @_QMcsPtest2(
  ! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>{{.*}}) {
  subroutine test2(a1, a2)
    type(t2) :: a1(:), a2(:)
    ! CHECK-DAG: %[[a1_decl:.*]]:2 = hlfir.declare %[[arg0]] {{.*}}
    ! CHECK-DAG: %[[a2_decl:.*]]:2 = hlfir.declare %[[arg1]] {{.*}}

    ! slice for a2%f1(1)
    ! CHECK: %[[a2_f1_1:.*]] = hlfir.designate %[[a2_decl]]#0{"f1"} {{.*}}(%c1)

    ! slice for a2%f1(5)
    ! CHECK: %[[a2_f1_5:.*]] = hlfir.designate %[[a2_decl]]#0{"f1"} {{.*}}(%c5{{.*}})

    ! slice for a2%f1(3)
    ! CHECK: %[[a2_f1_3:.*]] = hlfir.designate %[[a2_decl]]#0{"f1"} {{.*}}(%c3)

    ! RHS computation
    ! CHECK: %[[div_expr:.*]] = hlfir.elemental
    ! CHECK:   hlfir.designate %[[a2_f1_5]]
    ! CHECK:   hlfir.designate %[[a2_f1_3]]
    ! CHECK:   %[[val_f1_5:.*]] = fir.load {{.*}}
    ! CHECK:   %[[val_f1_3:.*]] = fir.load {{.*}}
    ! CHECK:   %[[div:.*]] = arith.divsi %[[val_f1_5]], %[[val_f1_3]]
    ! CHECK: %[[sum_expr:.*]] = hlfir.elemental
    ! CHECK:   hlfir.designate %[[a2_f1_1]]
    ! CHECK:   hlfir.apply %[[div_expr]]
    ! CHECK:   %[[sum:.*]] = arith.addi {{.*}}

    ! slice for a1%f2%d
    ! CHECK: %[[a1_f2:.*]] = hlfir.designate %[[a1_decl]]#0{"f2"}
    ! CHECK: %[[a1_f2_d:.*]] = hlfir.designate %[[a1_f2]]{"d"}

    ! Assignment
    ! CHECK: hlfir.assign %[[sum_expr]] to %[[a1_f2_d]]
    a1%f2%d = a2%f1(1) + a2%f1(5) / a2%f1(3)
  end subroutine test2

  ! CHECK-LABEL: func.func @_QMcsPtest3(
  ! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>{{.*}}) {
  subroutine test3(a3, a4)
    type(t3) :: a3(:), a4(:)
    ! CHECK-DAG: %[[a3_decl:.*]]:2 = hlfir.declare %[[arg0]] {{.*}}
    ! CHECK-DAG: %[[a4_decl:.*]]:2 = hlfir.declare %[[arg1]] {{.*}}

    ! Assignment 1
    ! a3%f(1,1)%f2%n = a4%f(2,2)%f1(4) - 4

    ! RHS: a4%f(2,2)%f1(4)
    ! CHECK: %[[a4_f:.*]] = hlfir.designate %[[a4_decl]]#0{"f"} {{.*}}(%c2, %c2
    ! CHECK: %[[a4_f1:.*]] = hlfir.designate %[[a4_f]]{"f1"} {{.*}}(%c4)

    ! CHECK: %[[sub_expr:.*]] = hlfir.elemental
    ! CHECK:   fir.load
    ! CHECK:   arith.subi

    ! LHS: a3%f(1,1)%f2%n
    ! CHECK: %[[a3_f:.*]] = hlfir.designate %[[a3_decl]]#0{"f"} {{.*}}(%c1, %c1
    ! CHECK: %[[a3_f2:.*]] = hlfir.designate %[[a3_f]]{"f2"}
    ! CHECK: %[[a3_n:.*]] = hlfir.designate %[[a3_f2]]{"n"}

    ! CHECK: hlfir.assign %[[sub_expr]] to %[[a3_n]]
    a3%f(1,1)%f2%n = a4%f(2,2)%f1(4) - 4

    ! Assignment 2
    ! a4%f(3,3)%f1(2) = a3%f(1,2)%f2%d + 4

    ! CHECK: %[[a3_f_rhs:.*]] = hlfir.designate %[[a3_decl]]#0{"f"} {{.*}}(%c1{{.*}}, %c2
    ! CHECK: %[[a3_rhs_f2:.*]] = hlfir.designate %[[a3_f_rhs]]{"f2"}
    ! CHECK: %[[a3_rhs_d:.*]] = hlfir.designate %[[a3_rhs_f2]]{"d"}
    ! CHECK: %[[add_expr:.*]] = hlfir.elemental
    ! CHECK:   fir.load
    ! CHECK:   arith.addi
    ! CHECK: %[[a4_f_lhs:.*]] = hlfir.designate %[[a4_decl]]#0{"f"} {{.*}}(%c3{{.*}}, %c3
    ! CHECK: %[[a4_f1_lhs:.*]] = hlfir.designate %[[a4_f_lhs]]{"f1"} {{.*}}(%c2
    ! CHECK: hlfir.assign %[[add_expr]] to %[[a4_f1_lhs]]
    a4%f(3,3)%f1(2) = a3%f(1,2)%f2%d + 4
  end subroutine test3
end module cs
