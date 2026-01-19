! Test lower of FORALL pointer assignment
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
! CHECK: %[[V_18:[0-9]+]] = fir.declare %arg0 dummy_scope %0 {{.*}} {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QFforallpolymorphic2Etar1"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>>, !fir.dscope) -> !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt{ptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFforallpolymorphic2Tdt>>>>}>>>>>
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


!! Test when LHS is unlimited polymorphic and RHS non-polymorphic intrinsic
!! type target.
! CHECK-LABEL: c.func @_QPforallpolymorphic3
subroutine forallPolymorphic3()
  TYPE :: DT
    CLASS(*), POINTER    :: Ptr => NULL()
  END TYPE

  TYPE(DT) :: D1(10)
  CHARACTER*1, TARGET :: TAR1(10)
  INTEGER :: I

  FORALL (I=1:10)
    D1(I)%Ptr => Tar1(I)
  END FORALL

! CHECK: %[[V_7:[0-9]+]] = fir.alloca !fir.array<10x!fir.type<_QFforallpolymorphic3Tdt{ptr:!fir.class<!fir.ptr<none>>}>> {bindc_name = "d1", uniq_name = "_QFforallpolymorphic3Ed1"}
! CHECK: %[[V_8:[0-9]+]] = fir.shape %c10 : (index) -> !fir.shape<1>
! CHECK: %[[V_9:[0-9]+]] = fir.declare %[[V_7]](%[[V_8]]) {uniq_name = "_QFforallpolymorphic3Ed1"} : (!fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphic3Tdt{ptr:!fir.class<!fir.ptr<none>>}>>>, !fir.shape<1>) -> !fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphic3Tdt{ptr:!fir.class<!fir.ptr<none>>}>>>
! CHECK: %[[V_16:[0-9]+]] = fir.alloca !fir.array<10x!fir.char<1>> {bindc_name = "tar1", fir.target, uniq_name = "_QFforallpolymorphic3Etar1"}
! CHECK: %[[V_17:[0-9]+]] = fir.declare %[[V_16]](%[[V_8]]) typeparams %c1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFforallpolymorphic3Etar1"} : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK: %[[V_24:[0-9]+]] = fir.convert %c1_i32 : (i32) -> index
! CHECK: %[[V_25:[0-9]+]] = fir.convert %c10_i32 : (i32) -> index
! CHECK: fir.do_loop %arg0 = %[[V_24]] to %[[V_25]] step %c1
! CHECK: {
! CHECK: %[[V_26:[0-9]+]] = fir.convert %arg0 : (index) -> i32
! CHECK: %[[V_27:[0-9]+]] = fir.convert %[[V_26]] : (i32) -> i64
! CHECK: %[[V_28:[0-9]+]] = fir.array_coor %[[V_9]](%[[V_8]]) %[[V_27]] : (!fir.ref<!fir.array<10x!fir.type<_QFforallpolymorphic3Tdt{ptr:!fir.class<!fir.ptr<none>>}>>>, !fir.shape<1>, i64) -> !fir.ref<!fir.type<_QFforallpolymorphic3Tdt{ptr:!fir.class<!fir.ptr<none>>}>>
! CHECK: %[[V_29:[0-9]+]] = fir.field_index ptr, !fir.type<_QFforallpolymorphic3Tdt{ptr:!fir.class<!fir.ptr<none>>}>
! CHECK: %[[V_30:[0-9]+]] = fir.coordinate_of %[[V_28]], ptr : (!fir.ref<!fir.type<_QFforallpolymorphic3Tdt{ptr:!fir.class<!fir.ptr<none>>}>>) -> !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[V_31:[0-9]+]] = fir.convert %[[V_26]] : (i32) -> i64
! CHECK: %[[V_32:[0-9]+]] = fir.array_coor %[[V_17]](%[[V_8]]) %31 : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, i64) -> !fir.ref<!fir.char<1>>
! CHECK: %[[V_33:[0-9]+]] = fir.embox %[[V_32]] : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.ptr<!fir.char<1>>>
! CHECK: %[[V_34:[0-9]+]] = fir.rebox %[[V_33]] : (!fir.box<!fir.ptr<!fir.char<1>>>) -> !fir.class<!fir.ptr<none>>
! CHECK: fir.store %[[V_34]] to %[[V_30]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: }

end subroutine forallPolymorphic3


!! Test the LHS of a pointer assignment gets the isPointer flag from the
!! RHS that is a reference to a function that returns a pointer.
! CHECK-LABEL: c.func @_QPforallpointerassignment1
  subroutine forallPointerAssignment1()
    type base
        real, pointer :: data => null()
    end type

    interface
      pure function makeData (i)
        real, pointer :: makeData
        integer*4, intent(in) :: i
      end function
    end interface

    type(base) :: co1(10)

    forall (i=1:10)
        co1(i)%data => makeData (i)
    end forall

! CHECK: %[[V_3:[0-9]+]] = fir.alloca i64
! CHECK: %[[V_3:[0-9]+]] = fir.alloca i32 {bindc_name = "i"}
! CHECK: %[[V_4:[0-9]+]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = ".result"}
! CHECK: %[[V_25:[0-9]+]] = fir.convert %c1_i32 : (i32) -> index
! CHECK: %[[V_26:[0-9]+]] = fir.convert %c10_i32 : (i32) -> index
! CHECK: %[[V_27:[0-9]+]] = fir.address_of(@{{_QQcl.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK: %[[V_28:[0-9]+]] = fir.convert %[[V_27]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK: %[[V_29:[0-9]+]] = fir.call @_FortranACreateDescriptorStack(%[[V_28]], %c{{.*}}) : (!fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
! CHECK: fir.do_loop %arg0 = %[[V_25]] to %[[V_26]] step %c1
! CHECK: {
! CHECK: %[[V_32:[0-9]+]] = fir.convert %arg0 : (index) -> i32
! CHECK: fir.store %[[V_32]] to %[[V_3]] : !fir.ref<i32>
! CHECK: %[[V_33:[0-9]+]] = fir.call @_QPmakedata(%[[V_3]]) proc_attrs<pure> fastmath<contract> : (!fir.ref<i32>) -> !fir.box<!fir.ptr<f32>>
! CHECK: fir.save_result %[[V_33]] to %[[V_4]] : !fir.box<!fir.ptr<f32>>, !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[V_34:[0-9]+]] = fir.declare %[[V_4]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[V_35:[0-9]+]] = fir.load %[[V_34]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK: %[[V_36:[0-9]+]] = fir.convert %[[V_35]] : (!fir.box<!fir.ptr<f32>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAPushDescriptor(%[[V_29]], %[[V_36]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
! CHECK: }

  end subroutine forallPointerAssignment1
