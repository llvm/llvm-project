! RUN: flang -fc1 -emit-fir -fwrapv %s -o - | FileCheck %s

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

    ! CHECK-DAG: %[[c_decl:.*]] = fir.declare %[[arg1]] {{.*}}
    ! CHECK-DAG: %[[c_rebox:.*]] = fir.rebox %[[c_decl]] : {{.*}}

    ! CHECK-DAG: %[[e_decl:.*]] = fir.declare %[[arg0]] {{.*}}
    ! CHECK-DAG: %[[e_rebox:.*]] = fir.rebox %[[e_decl]] : {{.*}}

    ! CHECK-DAG: %[[fldn_c:.*]] = fir.field_index n, !fir.type<_QMcsTr{n:i32,d:i32}>
    ! CHECK: %[[slice_c:.*]] = fir.slice {{.*}} path %[[fldn_c]] : (index, index, index, !fir.field) -> !fir.slice<1>
    ! CHECK: %[[c_n_rebox:.*]] = fir.rebox %[[c_rebox]] [%[[slice_c]]]
    ! CHECK-DAG: %[[fldn_e:.*]] = fir.field_index n, !fir.type<_QMcsTr{n:i32,d:i32}>
    ! CHECK: %[[slice_e:.*]] = fir.slice {{.*}} path %[[fldn_e]] : (index, index, index, !fir.field) -> !fir.slice<1>
    ! CHECK: %[[e_n_rebox:.*]] = fir.rebox %[[e_rebox]] [%[[slice_e]]]
    ! CHECK: fir.do_loop {{.*}} {
    ! CHECK:   %[[c_addr:.*]] = fir.array_coor %[[c_n_rebox]]
    ! CHECK:   %[[e_addr:.*]] = fir.array_coor %[[e_n_rebox]]
    ! CHECK:   %[[c_val:.*]] = fir.load %[[c_addr]]
    ! CHECK:   %[[e_val:.*]] = fir.load %[[e_addr]]
    ! CHECK:   %[[cmp:.*]] = arith.cmpi eq, %[[c_val]], %[[e_val]]
    ! CHECK:   {{.*}} = fir.convert %[[cmp]] : (i1) -> !fir.logical<4>
    ! CHECK:   fir.store {{.*}}
    ! CHECK: }
    c1 = all(c%n == e%n)
  end function c1

  ! CHECK-LABEL: func.func @_QMcsPtest2(
  ! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>{{.*}}) {
  subroutine test2(a1, a2)
    type(t2) :: a1(:), a2(:)
    ! CHECK-DAG: %[[a1_decl:.*]] = fir.declare %[[arg0]] {{.*}}
    ! CHECK-DAG: %[[a1_rebox:.*]] = fir.rebox %[[a1_decl]] : {{.*}}
    ! CHECK-DAG: %[[a2_decl:.*]] = fir.declare %[[arg1]] {{.*}}
    ! CHECK-DAG: %[[a2_rebox:.*]] = fir.rebox %[[a2_decl]] : {{.*}}

    ! slice for a2%f1(1)
    ! CHECK-DAG: %[[fld_f1_1:.*]] = fir.field_index f1, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
    ! CHECK: %[[slice_a2_f1_1:.*]] = fir.slice {{.*}} path %[[fld_f1_1]], {{.*}}
    ! CHECK: %[[a2_f1_1_rebox:.*]] = fir.rebox %[[a2_rebox]] [%[[slice_a2_f1_1]]]

    ! slice for a2%f1(5)
    ! CHECK-DAG: %[[fld_f1_5:.*]] = fir.field_index f1, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
    ! CHECK: %[[slice_a2_f1_5:.*]] = fir.slice {{.*}} path %[[fld_f1_5]], {{.*}}
    ! CHECK: %[[a2_f1_5_rebox:.*]] = fir.rebox %[[a2_rebox]] [%[[slice_a2_f1_5]]]

    ! slice for a2%f1(3)
    ! CHECK-DAG: %[[fld_f1_3:.*]] = fir.field_index f1, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
    ! CHECK: %[[slice_a2_f1_3:.*]] = fir.slice {{.*}} path %[[fld_f1_3]], {{.*}}
    ! CHECK: %[[a2_f1_3_rebox:.*]] = fir.rebox %[[a2_rebox]] [%[[slice_a2_f1_3]]]

    ! RHS computation
    ! CHECK: %[[temp:.*]] = fir.allocmem !fir.array<?xi32>
    ! CHECK: fir.do_loop {{.*}} {
    ! CHECK:   %[[val_f1_5:.*]] = fir.load {{.*}}
    ! CHECK:   %[[val_f1_3:.*]] = fir.load {{.*}}
    ! CHECK:   %[[div:.*]] = arith.divsi %[[val_f1_5]], %[[val_f1_3]]
    ! CHECK:   %[[val_f1_1:.*]] = fir.load {{.*}}
    ! CHECK:   %[[sum:.*]] = arith.addi %[[val_f1_1]], %[[div]]
    ! CHECK:   fir.store %[[sum]] to {{.*}}
    ! CHECK: }

    ! slice for a1%f2%d
    ! CHECK-DAG: %[[fld_f2:.*]] = fir.field_index f2, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
    ! CHECK: %[[slice_a1_f2:.*]] = fir.slice {{.*}} path %[[fld_f2]] : (index, index, index, !fir.field) -> !fir.slice<1>
    ! CHECK: %[[a1_f2_rebox:.*]] = fir.rebox %[[a1_rebox]] [%[[slice_a1_f2]]]
    ! CHECK-DAG: %[[fld_d:.*]] = fir.field_index d, !fir.type<_QMcsTr{n:i32,d:i32}>
    ! CHECK: %[[slice_d:.*]] = fir.slice {{.*}} path %[[fld_d]] : (index, index, index, !fir.field) -> !fir.slice<1>
    ! CHECK: %[[a1_f2_d_rebox:.*]] = fir.rebox %[[a1_f2_rebox]] [%[[slice_d]]]

    ! Assignment
    ! CHECK: fir.call @_FortranAAssign
    a1%f2%d = a2%f1(1) + a2%f1(5) / a2%f1(3)
  end subroutine test2

  ! CHECK-LABEL: func.func @_QMcsPtest3(
  ! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>{{.*}}) {
  subroutine test3(a3, a4)
    type(t3) :: a3(:), a4(:)
    ! CHECK-DAG: %[[a3_decl:.*]] = fir.declare %[[arg0]] {{.*}}
    ! CHECK-DAG: %[[a3_rebox:.*]] = fir.rebox %[[a3_decl]] : {{.*}}
    ! CHECK-DAG: %[[a4_decl:.*]] = fir.declare %[[arg1]] {{.*}}
    ! CHECK-DAG: %[[a4_rebox:.*]] = fir.rebox %[[a4_decl]] : {{.*}}

    ! Assignment 1
    ! a3%f(1,1)%f2%n = a4%f(2,2)%f1(4) - 4

    ! RHS: a4%f(2,2)%f1(4)
    ! CHECK: %[[slice_a4_f:.*]] = fir.slice {{.*}} path %{{.*}}, {{.*}}, {{.*}}
    ! CHECK: %[[a4_f_rebox:.*]] = fir.rebox %[[a4_rebox]] [%[[slice_a4_f]]]
    ! CHECK: %[[slice_a4_f1:.*]] = fir.slice {{.*}} path %{{.*}}, {{.*}}
    ! CHECK: %[[a4_f1_rebox:.*]] = fir.rebox %[[a4_f_rebox]] [%[[slice_a4_f1]]]

    ! CHECK: %[[temp1:.*]] = fir.allocmem !fir.array<?xi32>
    ! CHECK: fir.do_loop {{.*}} {
    ! CHECK:   fir.load
    ! CHECK:   arith.subi
    ! CHECK:   fir.store
    ! CHECK: }

    ! LHS: a3%f(1,1)%f2%n
    ! CHECK: %[[slice_a3_f:.*]] = fir.slice {{.*}} path %{{.*}}, {{.*}}, {{.*}}
    ! CHECK: %[[a3_f_rebox:.*]] = fir.rebox %[[a3_rebox]] [%[[slice_a3_f]]]
    ! CHECK: %[[slice_a3_f2:.*]] = fir.slice {{.*}} path %{{.*}}
    ! CHECK: %[[a3_f2_rebox:.*]] = fir.rebox %[[a3_f_rebox]] [%[[slice_a3_f2]]]
    ! CHECK: %[[slice_a3_n:.*]] = fir.slice {{.*}} path %{{.*}}
    ! CHECK: %[[a3_n_rebox:.*]] = fir.rebox %[[a3_f2_rebox]] [%[[slice_a3_n]]]

    ! CHECK: fir.call @_FortranAAssign
    a3%f(1,1)%f2%n = a4%f(2,2)%f1(4) - 4

    ! Assignment 2
    ! a4%f(3,3)%f1(2) = a3%f(1,2)%f2%d + 4

    ! CHECK: %[[temp2:.*]] = fir.allocmem !fir.array<?xi32>
    ! CHECK: fir.do_loop {{.*}} {
    ! CHECK:   fir.load
    ! CHECK:   arith.addi
    ! CHECK:   fir.store
    ! CHECK: }
    ! CHECK: fir.call @_FortranAAssign
    a4%f(3,3)%f1(2) = a3%f(1,2)%f2%d + 4
  end subroutine test3
end module cs
