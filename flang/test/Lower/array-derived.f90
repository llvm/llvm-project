! RUN: bbc %s -o - | FileCheck %s

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

  ! CHECK: func @_QMcsPc1(
  ! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>,
  ! CHECK-SAME: %[[arg1:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>)
  function c1(e, c)
    type(r), intent(in) :: e(:), c(:)
    ! CHECK-DAG: fir.alloca !fir.logical<1> {bindc_name = "c1", uniq_name = "_QMcsFc1Ec1"}
    logical*1 :: c1
    ! CHECK-DAG: %[[fldn:.*]] = fir.field_index n, !fir.type<_QMcsTr{n:i32,d:i32}>
    ! CHECK: %[[ext1:.*]]:3 = fir.box_dims %[[arg1]], %c0{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>, index) -> (index, index, index)
    ! CHECK-DAG: %[[slice1:.*]] = fir.slice %c1{{.*}}, %[[ext1]]#1, %c1{{.*}} path %[[fldn]] : (index, index, index, !fir.field) -> !fir.slice<1>
    ! CHECK-DAG: %[[ext0:.*]]:3 = fir.box_dims %[[arg0]], %c0{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>, index) -> (index, index, index)
    ! CHECK: %[[slice0:.*]] = fir.slice %c1{{.*}}, %[[ext0]]#1, %c1{{.*}} path %[[fldn]] : (index, index, index, !fir.field) -> !fir.slice<1>
    ! CHECK-DAG: = fir.array_coor %[[arg1]] [%[[slice1]]] %[[index:.*]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK-DAG: = fir.array_coor %[[arg0]] [%[[slice0]]] %[[index]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: = fir.call @_FortranAAll(
    c1 = all(c%n == e%n)
  end function c1

  ! CHECK: func @_QMcsPtest2(
  ! CHECK-SAME: %[[arg0:[^:]*]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>,
  ! CHECK-SAME:  %[[arg1:[^:]*]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>)
  subroutine test2(a1, a2)
    type(t2) :: a1(:), a2(:)
    ! CHECK-DAG: %[[VAL_33:.*]] = constant 1 : i64
    ! CHECK-DAG: %[[VAL_34:.*]] = constant 5 : i64
    ! CHECK-DAG: %[[VAL_35:.*]] = constant 3 : i64
    ! CHECK-DAG: %[[VAL_36:.*]] = constant 0 : index
    ! CHECK-DAG: %[[VAL_37:.*]] = constant 1 : index
    ! CHECK: %[[VAL_38:.*]] = fir.field_index f2, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
    ! CHECK: %[[VAL_39:.*]] = fir.field_index d, !fir.type<_QMcsTr{n:i32,d:i32}>
    ! CHECK: %[[VAL_40:.*]]:3 = fir.box_dims %[[arg0]], %[[VAL_36]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, index) -> (index, index, index)
    ! CHECK: %[[VAL_42:.*]] = fir.slice %[[VAL_37]], %[[VAL_40]]#1, %[[VAL_37]] path %[[VAL_38]], %[[VAL_39]] : (index, index, index, !fir.field, !fir.field) -> !fir.slice<1>
    ! CHECK: %[[VAL_43:.*]] = fir.field_index f1, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
    ! CHECK: %[[VAL_44:.*]]:3 = fir.box_dims %[[arg1]], %[[VAL_36]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, index) -> (index, index, index)
    ! CHECK: %[[VAL_46:.*]] = fir.slice %[[VAL_37]], %[[VAL_44]]#1, %[[VAL_37]] path %[[VAL_43]], %[[VAL_33]] : (index, index, index, !fir.field, i64) -> !fir.slice<1>
    ! CHECK: %[[VAL_47:.*]] = fir.slice %[[VAL_37]], %[[VAL_44]]#1, %[[VAL_37]] path %[[VAL_43]], %[[VAL_34]] : (index, index, index, !fir.field, i64) -> !fir.slice<1>
    ! CHECK: %[[VAL_48:.*]] = fir.slice %[[VAL_37]], %[[VAL_44]]#1, %[[VAL_37]] path %[[VAL_43]], %[[VAL_35]] : (index, index, index, !fir.field, i64) -> !fir.slice<1>
    ! CHECK: %[[CMP:.*]] = cmpi sgt, %[[VAL_40]]#1, %[[VAL_36]] : index 
    ! CHECK: %[[SELECT:.*]] = select %[[CMP]], %[[VAL_40]]#1, %[[VAL_36]] : index 
    ! CHECK: br ^bb1(%[[VAL_36]], %[[SELECT]] : index, index)
    ! CHECK: ^bb1(%[[VAL_49:.*]]: index, %[[VAL_50:.*]]: index):
    ! CHECK: %[[VAL_51:.*]] = cmpi sgt, %[[VAL_50]], %[[VAL_36]] : index
    ! CHECK: cond_br %[[VAL_51]], ^bb2, ^bb3
    ! CHECK: ^bb2:
    ! CHECK: %[[VAL_52:.*]] = addi %[[VAL_49]], %[[VAL_37]] : index
    ! CHECK: %[[VAL_53:.*]] = fir.array_coor %[[arg1]] {{\[}}%[[VAL_46]]] %[[VAL_52]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: %[[VAL_54:.*]] = fir.load %[[VAL_53]] : !fir.ref<i32>
    ! CHECK: %[[VAL_55:.*]] = fir.array_coor %[[arg1]] {{\[}}%[[VAL_47]]] %[[VAL_52]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: %[[VAL_56:.*]] = fir.load %[[VAL_55]] : !fir.ref<i32>
    ! CHECK: %[[VAL_57:.*]] = fir.array_coor %[[arg1]] {{\[}}%[[VAL_48]]] %[[VAL_52]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: %[[VAL_58:.*]] = fir.load %[[VAL_57]] : !fir.ref<i32>
    ! CHECK: %[[VAL_59:.*]] = divi_signed %[[VAL_56]], %[[VAL_58]] : i32
    ! CHECK: %[[VAL_60:.*]] = addi %[[VAL_54]], %[[VAL_59]] : i32
    ! CHECK: %[[VAL_61:.*]] = fir.array_coor %[[arg0]] {{\[}}%[[VAL_42]]] %[[VAL_52]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: fir.store %[[VAL_60]] to %[[VAL_61]] : !fir.ref<i32>
    ! CHECK: %[[VAL_62:.*]] = subi %[[VAL_50]], %[[VAL_37]] : index
    ! CHECK: br ^bb1(%[[VAL_52]], %[[VAL_62]] : index, index)
    ! CHECK: ^bb3:
    ! CHECK: return
    a1%f2%d = a2%f1(1) + a2%f1(5) / a2%f1(3)
  end subroutine test2

  ! CHECK: func @_QMcsPtest3(
  ! CHECK-SAME: %[[arg0:[^:]*]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>,
  ! CHECK-SAME:  %[[arg1:[^:]*]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>)
  subroutine test3(a3, a4)
    type(t3) :: a3(:), a4(:)
    ! CHECK-DAG: %[[VAL_63:.*]] = constant 4 : i64
    ! CHECK-DAG: %[[VAL_64:.*]] = constant 3 : i64
    ! CHECK-DAG: %[[VAL_65:.*]] = constant 1 : i64
    ! CHECK-DAG: %[[VAL_66:.*]] = constant 2 : i64
    ! CHECK-DAG: %[[VAL_67:.*]] = constant 4 : i32
    ! CHECK-DAG: %[[VAL_68:.*]] = constant 0 : index
    ! CHECK-DAG: %[[VAL_69:.*]] = constant 1 : index
    ! CHECK: %[[VAL_70:.*]] = fir.field_index f, !fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>
    ! CHECK: %[[VAL_71:.*]] = fir.field_index f2, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
    ! CHECK: %[[VAL_72:.*]] = fir.field_index n, !fir.type<_QMcsTr{n:i32,d:i32}>
    ! CHECK: %[[VAL_73:.*]]:3 = fir.box_dims %[[arg0]], %[[VAL_68]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, index) -> (index, index, index)
    ! CHECK: %[[VAL_75:.*]] = fir.slice %[[VAL_69]], %[[VAL_73]]#1, %[[VAL_69]] path %[[VAL_70]], %[[VAL_65]], %[[VAL_65]], %[[VAL_71]], %[[VAL_72]] : (index, index, index, !fir.field, i64, i64, !fir.field, !fir.field) -> !fir.slice<1>
    ! CHECK: %[[VAL_76:.*]] = fir.field_index f1, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
    ! CHECK: %[[VAL_77:.*]]:3 = fir.box_dims %[[arg1]], %[[VAL_68]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, index) -> (index, index, index)
    ! CHECK: %[[VAL_79:.*]] = fir.slice %[[VAL_69]], %[[VAL_77]]#1, %[[VAL_69]] path %[[VAL_70]], %[[VAL_66]], %[[VAL_66]], %[[VAL_76]], %[[VAL_63]] : (index, index, index, !fir.field, i64, i64, !fir.field, i64) -> !fir.slice<1>
    ! CHECK: %[[CMP:.*]] = cmpi sgt, %[[VAL_73]]#1, %[[VAL_68]] : index 
    ! CHECK: %[[SELECT:.*]] = select %[[CMP]], %[[VAL_73]]#1, %[[VAL_68]] : index 
    ! CHECK: br ^bb1(%[[VAL_68]], %[[SELECT]] : index, index)
    ! CHECK: ^bb1(%[[VAL_80:.*]]: index, %[[VAL_81:.*]]: index):
    ! CHECK: %[[VAL_82:.*]] = cmpi sgt, %[[VAL_81]], %[[VAL_68]] : index
    ! CHECK: cond_br %[[VAL_82]], ^bb2, ^bb3
    ! CHECK: ^bb2:
    ! CHECK: %[[VAL_83:.*]] = addi %[[VAL_80]], %[[VAL_69]] : index
    ! CHECK: %[[VAL_84:.*]] = fir.array_coor %[[arg1]] {{\[}}%[[VAL_79]]] %[[VAL_83]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: %[[VAL_85:.*]] = fir.load %[[VAL_84]] : !fir.ref<i32>
    ! CHECK: %[[VAL_86:.*]] = subi %[[VAL_85]], %[[VAL_67]] : i32
    ! CHECK: %[[VAL_87:.*]] = fir.array_coor %[[arg0]] {{\[}}%[[VAL_75]]] %[[VAL_83]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: fir.store %[[VAL_86]] to %[[VAL_87]] : !fir.ref<i32>
    ! CHECK: %[[VAL_88:.*]] = subi %[[VAL_81]], %[[VAL_69]] : index
    ! CHECK: br ^bb1(%[[VAL_83]], %[[VAL_88]] : index, index)
    ! CHECK: ^bb3:
    ! CHECK: %[[VAL_89:.*]] = fir.slice %[[VAL_69]], %[[VAL_77]]#1, %[[VAL_69]] path %[[VAL_70]], %[[VAL_64]], %[[VAL_64]], %[[VAL_76]], %[[VAL_66]] : (index, index, index, !fir.field, i64, i64, !fir.field, i64) -> !fir.slice<1>
    ! CHECK: %[[VAL_90:.*]] = fir.field_index d, !fir.type<_QMcsTr{n:i32,d:i32}>
    ! CHECK: %[[VAL_91:.*]] = fir.slice %[[VAL_69]], %[[VAL_73]]#1, %[[VAL_69]] path %[[VAL_70]], %[[VAL_65]], %[[VAL_66]], %[[VAL_71]], %[[VAL_90]] : (index, index, index, !fir.field, i64, i64, !fir.field, !fir.field) -> !fir.slice<1>
    ! CHECK: %[[CMP2:.*]] = cmpi sgt, %[[VAL_77]]#1, %[[VAL_68]] : index 
    ! CHECK: %[[SELECT2:.*]] = select %[[CMP2]], %[[VAL_77]]#1, %[[VAL_68]] : index 
    ! CHECK: br ^bb4(%[[VAL_68]], %[[SELECT2]] : index, index)
    ! CHECK: ^bb4(%[[VAL_92:.*]]: index, %[[VAL_93:.*]]: index):
    ! CHECK: %[[VAL_94:.*]] = cmpi sgt, %[[VAL_93]], %[[VAL_68]] : index
    ! CHECK: cond_br %[[VAL_94]], ^bb5, ^bb6
    ! CHECK: ^bb5:
    ! CHECK: %[[VAL_95:.*]] = addi %[[VAL_92]], %[[VAL_69]] : index
    ! CHECK: %[[VAL_96:.*]] = fir.array_coor %[[arg0]] {{\[}}%[[VAL_91]]] %[[VAL_95]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: %[[VAL_97:.*]] = fir.load %[[VAL_96]] : !fir.ref<i32>
    ! CHECK: %[[VAL_98:.*]] = addi %[[VAL_97]], %[[VAL_67]] : i32
    ! CHECK: %[[VAL_99:.*]] = fir.array_coor %[[arg1]] {{\[}}%[[VAL_89]]] %[[VAL_95]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: fir.store %[[VAL_98]] to %[[VAL_99]] : !fir.ref<i32>
    ! CHECK: %[[VAL_100:.*]] = subi %[[VAL_93]], %[[VAL_69]] : index
    ! CHECK: br ^bb4(%[[VAL_95]], %[[VAL_100]] : index, index)
    ! CHECK: ^bb6:
    ! CHECK: return
    a3%f(1,1)%f2%n = a4%f(2,2)%f1(4) - 4
    a4%f(3,3)%f1(2) = a3%f(1,2)%f2%d + 4
  end subroutine test3
end module cs

! CHECK: func private @_FortranAAll(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1 attributes {fir.runtime}
