! RUN: bbc %s -o - | FileCheck %s

module components_test
  type t1
     integer :: i(6)
     real :: r(5)
  end type t1

  type t2
     type(t1) :: g1(3,3), g2(4,4,4)
     integer :: g3(5)
  end type t2

  type t3
     type(t1) :: h1(3)
     type(t2) :: h2(4)
  end type t3

  type(t3) :: instance

contains

  ! CHECK-LABEL: func @_QMcomponents_testPs1(
  subroutine s1(i,j)
    ! CHECK-DAG: %[[VAL_0:.*]] = arith.constant 1 : i32
    ! CHECK-DAG: %[[VAL_1:.*]] = arith.constant 6 : i32
    ! CHECK-DAG: %[[VAL_2:.*]] = arith.constant 0 : i64
    ! CHECK-DAG: %[[VAL_3:.*]] = arith.constant 1 : i64
    ! CHECK-DAG: %[[VAL_4:.*]] = arith.constant 2 : i64
    ! CHECK: %[[VAL_5:.*]] = fir.address_of(@_QMcomponents_testEinstance) : !fir.ref<!fir.type<_QMcomponents_testTt3{h1:!fir.array<3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,h2:!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>}>>
    ! CHECK: %[[VAL_6:.*]] = fir.load %[[VAL_7:.*]] : !fir.ref<i32>
    ! CHECK: %[[VAL_8:.*]] = arith.cmpi sge, %[[VAL_6]], %[[VAL_0]] : i32
    ! CHECK: %[[VAL_9:.*]] = arith.cmpi sle, %[[VAL_6]], %[[VAL_1]] : i32
    ! CHECK: %[[VAL_10:.*]] = arith.andi %[[VAL_8]], %[[VAL_9]] : i1
    ! CHECK: cond_br %[[VAL_10]], ^bb1, ^bb2
    ! CHECK: ^bb1:
    ! CHECK: %[[VAL_11:.*]] = fir.field_index h2, !fir.type<_QMcomponents_testTt3{h1:!fir.array<3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,h2:!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>}>
    ! CHECK: %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_11]] : (!fir.ref<!fir.type<_QMcomponents_testTt3{h1:!fir.array<3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,h2:!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>}>>, !fir.field) -> !fir.ref<!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>>
    ! CHECK: %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_3]] : (!fir.ref<!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>>, i64) -> !fir.ref<!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>
    ! CHECK: %[[VAL_14:.*]] = fir.field_index g2, !fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>
    ! CHECK: %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_14]] : (!fir.ref<!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>>
    ! CHECK: %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_15]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (!fir.ref<!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>>, i64, i64, i64) -> !fir.ref<!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>
    ! CHECK: %[[VAL_17:.*]] = fir.field_index i, !fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>
    ! CHECK: %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_16]], %[[VAL_17]] : (!fir.ref<!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>, !fir.field) -> !fir.ref<!fir.array<6xi32>>
    ! CHECK: %[[VAL_19:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
    ! CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
    ! CHECK: %[[VAL_20_ADJ:.*]] = arith.subi %[[VAL_20]], %[[VAL_3]] : i64
    ! CHECK: %[[VAL_21:.*]] = fir.coordinate_of %[[VAL_18]], %[[VAL_20_ADJ]] : (!fir.ref<!fir.array<6xi32>>, i64) -> !fir.ref<i32>
    ! CHECK: %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
    ! CHECK: fir.store %[[VAL_22]] to %[[VAL_23:.*]] : !fir.ref<i32>
    ! CHECK: br ^bb2
    ! CHECK: ^bb2:
    ! CHECK: return
    if (j >= 1 .and. j <= 6) then
       i = instance%h2(2)%g2(1,2,3)%i(j)
    end if
  end subroutine s1

end module components_test

! CHECK-LABEL: func @_QPsliced_base() {
! CHECK-DAG:     %[[VAL_0:.*]] = arith.constant 50 : index
! CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 42 : i32
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.array<100x!fir.type<_QFsliced_baseTt{x:f32,y:i32}>> {bindc_name = "a", uniq_name = "_QFsliced_baseEa"}
! CHECK:         %[[VAL_6:.*]] = fir.field_index y, !fir.type<_QFsliced_baseTt{x:f32,y:i32}>
! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_8:.*]] = fir.slice %[[VAL_1]], %[[VAL_0]], %[[VAL_1]] path %[[VAL_6]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:         br ^bb1(%[[VAL_3]], %[[VAL_0]] : index, index)
! CHECK:       ^bb1(%[[VAL_9:.*]]: index, %[[VAL_10:.*]]: index):
! CHECK:         %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_3]] : index
! CHECK:         cond_br %[[VAL_11]], ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_12:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
! CHECK:         %[[VAL_13:.*]] = fir.array_coor %[[VAL_5]](%[[VAL_7]]) {{\[}}%[[VAL_8]]] %[[VAL_12]] : (!fir.ref<!fir.array<100x!fir.type<_QFsliced_baseTt{x:f32,y:i32}>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_2]] to %[[VAL_13]] : !fir.ref<i32>
! CHECK:         %[[VAL_14:.*]] = arith.subi %[[VAL_10]], %[[VAL_1]] : index
! CHECK:         br ^bb1(%[[VAL_12]], %[[VAL_14]] : index, index)
! CHECK:       ^bb3:
! CHECK:         %[[VAL_15:.*]] = fir.embox %[[VAL_5]](%[[VAL_7]]) {{\[}}%[[VAL_8]]] : (!fir.ref<!fir.array<100x!fir.type<_QFsliced_baseTt{x:f32,y:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<50xi32>>
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<50xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         fir.call @_QPtakes_int_array(%[[VAL_16]]) {{.*}}: (!fir.box<!fir.array<?xi32>>) -> ()
! CHECK:         return
! CHECK:       }

subroutine sliced_base()
  interface
    subroutine takes_int_array(i)
      integer :: i(:)
    end subroutine
  end interface
  type t
    real :: x
    integer :: y
  end type
  type(t) :: a(100)
  a(1:50)%y = 42
  call takes_int_array(a(1:50)%y)
end subroutine

! CHECK-LABEL: issue772
subroutine issue772(a, x)
  ! Verify that sub-expressions inside a component reference are
  ! only evaluated once.
  type t
    real :: b(100)
  end type
  real :: x(100)
  type(t) :: a(100)
  ! CHECK: fir.call @_QPifoo()
  ! CHECK-NOT: fir.call @_QPifoo()
  x = a(ifoo())%b(1:100:1)
  ! CHECK: fir.call @_QPibar()
  ! CHECK-NOT: fir.call @_QPibar()
  print *, a(20)%b(1:ibar():1)
  ! CHECK return
end subroutine

! -----------------------------------------------------------------------------
!     Test array%character array sections
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPlhs_char_section(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFlhs_char_sectionTt{c:!fir.char<1,5>}>>>{{.*}}) {
subroutine lhs_char_section(a)
  ! CHECK-DAG: %[[VAL_1:.*]] = arith.constant 5 : index
  ! CHECK-DAG: %[[VAL_2:.*]] = arith.constant false
  ! CHECK-DAG: %[[VAL_3:.*]] = arith.constant 10 : index
  ! CHECK-DAG: %[[VAL_4:.*]] = arith.constant 0 : index
  ! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_6:.*]] = fir.field_index c, !fir.type<_QFlhs_char_sectionTt{c:!fir.char<1,5>}>
  ! CHECK: %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_8:.*]] = fir.slice %[[VAL_5]], %[[VAL_3]], %[[VAL_5]] path %[[VAL_6]] : (index, index, index, !fir.field) -> !fir.slice<1>
  ! CHECK: %[[VAL_9:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,5>>
  ! CHECK: br ^bb1(%[[VAL_4]], %[[VAL_3]] : index, index)
  ! CHECK: ^bb1(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
  ! CHECK: %[[VAL_12:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_4]] : index
  ! CHECK: cond_br %[[VAL_12]], ^bb2, ^bb3
  ! CHECK: ^bb2:
  ! CHECK: %[[VAL_13:.*]] = arith.addi %[[VAL_10]], %[[VAL_5]] : index
  ! CHECK: %[[VAL_14:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) {{\[}}%[[VAL_8]]] %[[VAL_13]] : (!fir.ref<!fir.array<10x!fir.type<_QFlhs_char_sectionTt{c:!fir.char<1,5>}>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,5>>
  ! CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_1]] : (index) -> i64
  ! CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_17:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_16]], %[[VAL_17]], %[[VAL_15]], %[[VAL_2]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_18:.*]] = arith.subi %[[VAL_11]], %[[VAL_5]] : index
  ! CHECK: br ^bb1(%[[VAL_13]], %[[VAL_18]] : index, index)

  type t
   character(5) :: c
  end type
  type(t) :: a(10)
  a%c = "hello"
    ! CHECK: return
    ! CHECK: }
end subroutine

! CHECK-LABEL: func @_QPrhs_char_section(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QFrhs_char_sectionTt{c:!fir.char<1,10>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine rhs_char_section(a, c)
  ! CHECK-DAG: %[[VAL_2:.*]] = arith.constant false
  ! CHECK-DAG: %[[VAL_3:.*]] = arith.constant 10 : index
  ! CHECK-DAG: %[[VAL_4:.*]] = arith.constant 0 : index
  ! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
  ! CHECK: %[[VAL_8:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_9:.*]] = fir.field_index c, !fir.type<_QFrhs_char_sectionTt{c:!fir.char<1,10>}>
  ! CHECK: %[[VAL_10:.*]] = fir.slice %[[VAL_5]], %[[VAL_3]], %[[VAL_5]] path %[[VAL_9]] : (index, index, index, !fir.field) -> !fir.slice<1>
  ! CHECK: br ^bb1(%[[VAL_4]], %[[VAL_3]] : index, index)
  ! CHECK: ^bb1(%[[VAL_11:.*]]: index, %[[VAL_12:.*]]: index):
  ! CHECK: %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_4]] : index
  ! CHECK: cond_br %[[VAL_13]], ^bb2, ^bb3
  ! CHECK: ^bb2:
  ! CHECK: %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_5]] : index
  ! CHECK: %[[VAL_15:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_8]]) {{\[}}%[[VAL_10]]] %[[VAL_14]] : (!fir.ref<!fir.array<10x!fir.type<_QFrhs_char_sectionTt{c:!fir.char<1,10>}>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_16:.*]] = fir.array_coor %[[VAL_7]](%[[VAL_8]]) %[[VAL_14]] : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_17:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
  ! CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_18]], %[[VAL_19]], %[[VAL_17]], %[[VAL_2]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_20:.*]] = arith.subi %[[VAL_12]], %[[VAL_5]] : index
  ! CHECK: br ^bb1(%[[VAL_14]], %[[VAL_20]] : index, index)

  type t
   character(10) :: c
  end type
  type(t) :: a(10)
  character(10) :: c(10)
  c = a%c
    ! CHECK: return
    ! CHECK: }
end subroutine

! CHECK-LABEL: func @_QPelemental_char_section(
! CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.array<{{.*}}>>{{.*}}, %[[I:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
subroutine elemental_char_section(a, i)
  type t
   character(10) :: c
  end type
  type(t) :: a(10)
  integer :: i(10)
  ! CHECK-DAG:  %[[VAL_34:.*]] = arith.constant 5 : index
  ! CHECK-DAG:  %[[VAL_35:.*]] = arith.constant false
  ! CHECK-DAG:  %[[VAL_36:.*]] = arith.constant 10 : index
  ! CHECK-DAG:  %[[VAL_37:.*]] = arith.constant 0 : index
  ! CHECK-DAG:  %[[VAL_38:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_39:.*]] = fir.shape %[[VAL_36]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_40:.*]] = fir.field_index c, !fir.type<_QFelemental_char_sectionTt{c:!fir.char<1,10>}>
  ! CHECK: %[[VAL_41:.*]] = fir.slice %[[VAL_38]], %[[VAL_36]], %[[VAL_38]] path %[[VAL_40]] : (index, index, index, !fir.field) -> !fir.slice<1>
  ! CHECK: %[[VAL_42:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,5>>
  ! CHECK: br ^bb1(%[[VAL_37]], %[[VAL_36]] : index, index)
  ! CHECK:^bb1(%[[VAL_43:.*]]: index, %[[VAL_44:.*]]: index):
  ! CHECK: %[[VAL_45:.*]] = arith.cmpi sgt, %[[VAL_44]], %[[VAL_37]] : index
  ! CHECK: cond_br %[[VAL_45]], ^bb2, ^bb3
  ! CHECK:^bb2:
  ! CHECK: %[[VAL_46:.*]] = arith.addi %[[VAL_43]], %[[VAL_38]] : index
  ! CHECK: %[[VAL_47:.*]] = fir.array_coor %[[A]](%[[VAL_39]]) {{\[}}%[[VAL_41]]] %[[VAL_46]] : (!fir.ref<!fir.array<10x!fir.type<_QFelemental_char_sectionTt{c:!fir.char<1,10>}>>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_50:.*]] = fir.convert %[[VAL_36]] : (index) -> i64
  ! CHECK: %[[VAL_51:.*]] = fir.convert %[[VAL_42]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_52:.*]] = fir.convert %[[VAL_34]] : (index) -> i64
  ! CHECK: %[[VAL_53:.*]] = fir.call @_FortranAScan1(%[[VAL_49]], %[[VAL_50]], %[[VAL_51]], %[[VAL_52]], %[[VAL_35]]) {{.*}}: (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
  ! CHECK: %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i64) -> i32
  ! CHECK: %[[VAL_55:.*]] = fir.array_coor %[[I]](%[[VAL_39]]) %[[VAL_46]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  ! CHECK: fir.store %[[VAL_54]] to %[[VAL_55]] : !fir.ref<i32>
  ! CHECK: %[[VAL_57:.*]] = arith.subi %[[VAL_44]], %[[VAL_38]] : index
  ! CHECK: br ^bb1(%[[VAL_46]], %[[VAL_57]] : index, index)
  i = scan(a%c, "hello")
end subroutine

! CHECK-LABEL: extended_type_components
subroutine extended_type_components
  type t1
    integer :: t1i = 1
  end type t1
  type, extends(t1) :: t2
    integer :: t2i = 2
  end type t2
  type, extends(t2) :: t3
    integer :: t3i = 3
  end type t3
  type, extends(t3) :: t4
    integer :: t4i = 4
  end type t4

  type u1
    integer :: u1i = 11
  end type u1
  type, extends(u1) :: u2
    integer :: u2i = 22
    type(t3) :: u2t3
    type(t3) :: u2t4
  end type u2
  type, extends(u2) :: u3
    integer :: u3i = 33
  end type u3

  ! CHECK: %[[u3v:.*]] = fir.alloca !fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}> {bindc_name = "u3v", uniq_name = "_QFextended_type_componentsEu3v"}
  type(u3) :: u3v
  ! CHECK: %[[u3va:.*]] = fir.alloca !fir.array<5x!fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>> {bindc_name = "u3va", uniq_name = "_QFextended_type_componentsEu3va"}
  type(u3) :: u3va(5)

  ! CHECK: %[[VAL_10:.*]] = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_11:.*]] = fir.field_index u2t3, !fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>
  ! CHECK: %[[VAL_12:.*]] = fir.field_index t1i, !fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>
  ! CHECK: %[[VAL_13:.*]] = fir.coordinate_of %[[u3v]], %[[VAL_11]], %[[VAL_12]] : (!fir.ref<!fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>>, !fir.field, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
  ! CHECK: %[[VAL_15:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_10]], %[[VAL_14]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print*, u3v%u2t3%t1i

  ! CHECK: %[[VAL_20:.*]] = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_21:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
  ! CHECK: %[[VAL_22:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_20]], %[[VAL_21]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print*, u3v%u2%u2t3%t2%t1%t1i ! different syntax for the previous value

  ! CHECK: %[[VAL_30:.*]] = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_31:.*]] = fir.field_index u2t4, !fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>
  ! CHECK: %[[VAL_32:.*]] = fir.coordinate_of %[[u3v]], %[[VAL_31]], %[[VAL_12]] : (!fir.ref<!fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>>, !fir.field, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[VAL_33:.*]] = fir.load %[[VAL_32]] : !fir.ref<i32>
  ! CHECK: %[[VAL_34:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_30]], %[[VAL_33]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print*, u3v%u2t4%t1i

  ! CHECK: %[[VAL_40:.*]] = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_41:.*]] = fir.field_index t2i, !fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>
  ! CHECK: %[[VAL_42:.*]] = fir.coordinate_of %[[u3v]], %[[VAL_31]], %[[VAL_41]] : (!fir.ref<!fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>>, !fir.field, !fir.field) -> !fir.ref<i32>
  ! CHECK: %[[VAL_43:.*]] = fir.load %[[VAL_42]] : !fir.ref<i32>
  ! CHECK: %[[VAL_44:.*]] = fir.call @_FortranAioOutputInteger32(%[[VAL_40]], %[[VAL_43]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
  print*, u3v%u2t4%t2i

  ! CHECK: %[[VAL_50:.*]] = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_51:.*]] = fir.slice %c1{{.*}}, %c5{{.*}}, %c1{{.*}} path %{{.*}}, %{{.*}} : (index, index, index, !fir.field, !fir.field) -> !fir.slice<1>
  ! CHECK: %[[VAL_52:.*]] = fir.embox %[[u3va]](%{{.*}}) [%[[VAL_51]]] : (!fir.ref<!fir.array<5x!fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>>>,
  ! CHECK: %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
  ! CHECK: %[[VAL_54:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_50]], %[[VAL_53]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  print*, u3va%u2t3%t1i

  ! CHECK: %[[VAL_60:.*]] = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_61:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_60]], %[[VAL_53]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  print*, u3va%u2%u2t3%t2%t1%t1i ! different syntax for the previous value

  ! CHECK: %[[VAL_70:.*]] = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_71:.*]] = fir.slice %c1{{.*}}, %c5{{.*}}, %c1{{.*}} path %{{.*}}, %{{.*}} : (index, index, index, !fir.field, !fir.field) -> !fir.slice<1>
  ! CHECK: %[[VAL_72:.*]] = fir.embox %[[u3va]](%{{.*}}) [%[[VAL_71]]] : (!fir.ref<!fir.array<5x!fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<5xi32>>
  ! CHECK: %[[VAL_73:.*]] = fir.convert %[[VAL_72]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
  ! CHECK: %[[VAL_74:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_70]], %[[VAL_73]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  print*, u3va%u2t4%t1i

  ! CHECK: %[[VAL_80:.*]] = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_81:.*]] = fir.slice %c1{{.*}}, %c5{{.*}}, %c1{{.*}} path %{{.*}}, %{{.*}} : (index, index, index, !fir.field, !fir.field) -> !fir.slice<1>
  ! CHECK: %[[VAL_82:.*]] = fir.embox %[[u3va]](%{{.*}}) [%[[VAL_81]]] : (!fir.ref<!fir.array<5x!fir.type<_QFextended_type_componentsTu3{u1i:i32,u2i:i32,u2t3:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u2t4:!fir.type<_QFextended_type_componentsTt3{t1i:i32,t2i:i32,t3i:i32}>,u3i:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<5xi32>>
  ! CHECK: %[[VAL_83:.*]] = fir.convert %[[VAL_82]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
  ! CHECK: %[[VAL_84:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_80]], %[[VAL_83]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  print*, u3va%u2t4%t2i
  end subroutine extended_type_components
