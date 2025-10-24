! Test lower of FORALL polymorphic pointer assignment 
! RUN: bbc -emit-fir %s -o - | FileCheck %s

!! Test when LHS is polymorphic and RHS is not polymorphic
! CHECK-LABEL: c.func @_QPforallpolymorphic
  subroutine forallPolymorphic()
  TYPE :: DT
    CLASS(DT), POINTER    :: Ptr(:) => NULL()
  END TYPE

  TYPE, EXTENDS(DT) :: DT1
  END TYPE

  TYPE(DT1), TARGET  :: Tar1(10)
  CLASS(DT), POINTER :: T(:)
  integer :: I

  FORALL (I=1:10)
    T(I)%Ptr => Tar1
  END FORALL

! CHECK: %[[V_11:[0-9]+]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>> {bindc_name = "t", uniq_name = "_QFforallpolymorphicEt"}
! CHECK: %[[V_15:[0-9]+]] = fir.declare %[[V_11]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFforallpolymorphicEt"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>>) -> !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>>
! CHECK: %[[V_16:[0-9]+]] = fir.alloca !fir.array<10x!fir.type<_QFforallpolymorphicTdt1{dt:!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>}>> {bindc_name = "tar1", fir.target, uniq_name = "_QFforallpolymorphicEtar1"}
! CHECK: %[[V_17:[0-9]+]] = fir.shape %c10 : (index) -> !fir.shape<1>
! CHECK: %[[V_18:[0-9]+]] = fir.declare %[[V_16]](%[[V_17]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFforallpolymorphicEtar1"} : (!fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphicTdt1{dt:!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphicTdt1{dt:!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>}>>>
! CHECK: %[[V_19:[0-9]+]] = fir.embox %[[V_18]](%[[V_17]]) : (!fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphicTdt1{dt:!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<_QFforallpolymorphicTdt1{dt:!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>}>>>
! CHECK: %[[V_34:[0-9]+]] = fir.convert %c1_i32 : (i32) -> index
! CHECK: %[[V_35:[0-9]+]] = fir.convert %c10_i32 : (i32) -> index
! CHECK: fir.do_loop %arg0 = %[[V_34]] to %[[V_35]] step %c1
! CHECK: {
! CHECK: %[[V_36:[0-9]+]] = fir.convert %arg0 : (index) -> i32
! CHECK: %[[V_37:[0-9]+]] = fir.load %[[V_15]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>>
! CHECK: %[[V_38:[0-9]+]] = fir.convert %[[V_36]] : (i32) -> i64
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[V_39:[0-9]+]]:3 = fir.box_dims %37, %[[C0]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>, index) -> (index, index, index)
! CHECK: %[[V_40:[0-9]+]] = fir.shift %[[V_39]]#0 : (index) -> !fir.shift<1>
! CHECK: %[[V_41:[0-9]+]] = fir.array_coor %[[V_37]](%[[V_40]]) %[[V_38]] : (!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>, !fir.shift<1>, i64) -> !fir.ref<!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>
! CHECK: %[[V_42:[0-9]+]] = fir.embox %[[V_41]] source_box %[[V_37]] : (!fir.ref<!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>, !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>) -> !fir.class<!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>
! CHECK: %[[V_43:[0-9]+]] = fir.field_index ptr, !fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>
! CHECK: %[[V_44:[0-9]+]] = fir.coordinate_of %[[V_42]], ptr : (!fir.class<!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>) -> !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>>
! CHECK: %[[V_45:[0-9]+]] = fir.embox %[[V_18]](%[[V_17]]) : (!fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphicTdt1{dt:!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>}>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<10x!fir.type<_QFforallpolymorphicTdt1{dt:!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>}>>>>
! CHECK: %[[V_46:[0-9]+]] = fir.convert %[[V_45]] : (!fir.box<!fir.ptr<!fir.array<10x!fir.type<_QFforallpolymorphicTdt1{dt:!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>}>>>>) -> !fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>
! CHECK: fir.store %[[V_46]] to %[[V_44]] : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt{ptr:!fir.class<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphicTdt>>>>}>>>>>
! CHECK: }

  end subroutine forallPolymorphic

!! Test when LHS is not polymorphic but RHS is polymorphic
! CHECK-LABEL: c.func @_QPforallpolymorphic2(
! CHECK-SAME: %arg0: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>> {fir.bindc_name = "tar1", fir.target}) {
  subroutine forallPolymorphic2(Tar1)
  TYPE :: DT
    TYPE(DT), POINTER    :: Ptr(:) => NULL()
  END TYPE

  TYPE, EXTENDS(DT) :: DT1
  END TYPE

  CLASS(DT), ALLOCATABLE, TARGET  :: Tar1(:)
  TYPE(DT) :: T(10)
  integer :: I

  FORALL (I=1:10)
    T(I)%Ptr => Tar1
  END FORALL

! CHECK: %[[V_11:[0-9]+]] = fir.alloca !fir.array<10x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>> {bindc_name = "t", uniq_name = "_QFforallpolymorphic2Et"}
! CHECK: %[[V_12:[0-9]+]] = fir.shape %c10 : (index) -> !fir.shape<1>
! CHECK: %[[V_13:[0-9]+]] = fir.declare %[[V_11]](%[[V_12]]) {uniq_name = "_QFforallpolymorphic2Et"} : (!fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>
! CHECK: %[[V_18:[0-9]+]] = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFforallpolymorphic2Etar1"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>>, !fir.dscope) -> !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>>
! CHECK: %[[V_30:[0-9]+]] = fir.convert %c1_i32 : (i32) -> index
! CHECK: %[[V_31:[0-9]+]] = fir.convert %c10_i32 : (i32) -> index
! CHECK: fir.do_loop %arg1 = %[[V_30]] to %[[V_31]] step %c1
! CHECK: {
! CHECK: %[[V_32:[0-9]+]] = fir.convert %arg1 : (index) -> i32
! CHECK: %[[V_33:[0-9]+]] = fir.convert %[[V_32]] : (i32) -> i64
! CHECK: %[[V_34:[0-9]+]] = fir.array_coor %[[V_13]](%[[V_12]]) %[[V_33]] : (!fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>, !fir.shape<1>, i64) -> !fir.ref<!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>
! CHECK: %[[V_35:[0-9]+]] = fir.field_index ptr, !fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>
! CHECK: %[[V_36:[0-9]+]] = fir.coordinate_of %[[V_34]], ptr : (!fir.ref<!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>>
! CHECK: %[[V_37:[0-9]+]] = fir.load %[[V_18]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>>
! CHECK: %[[V_38:[0-9]+]]:3 = fir.box_dims %[[V_37]], %c0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>, index) -> (index, index, index)
! CHECK: %[[V_39:[0-9]+]] = fir.shift %[[V_38]]#0 : (index) -> !fir.shift<1>
! CHECK: %[[V_40:[0-9]+]] = fir.rebox %[[V_37]](%[[V_39]]) : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>
! CHECK: fir.store %[[V_40]] to %[[V_36]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>>
! CHECK: }

  end subroutine forallPolymorphic2

