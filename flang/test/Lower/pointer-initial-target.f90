! Test lowering of pointer initial target
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! -----------------------------------------------------------------------------
!     Test scalar initial data target that are simple names
! -----------------------------------------------------------------------------

subroutine scalar()
  real, save, target :: x
  real, pointer :: p => x
! CHECK-LABEL: fir.global internal @_QFscalarEp : !fir.box<!fir.ptr<f32>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFscalarEx) : !fir.ref<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]] : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<f32>>
end subroutine

subroutine scalar_char()
  character(10), save, target :: x
  character(:), pointer :: p => x
! CHECK-LABEL: fir.global internal @_QFscalar_charEp : !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFscalar_charEx) : !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[xCast:.*]] = fir.convert %[[x]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ptr<!fir.char<1,?>>
  ! CHECK: %[[box:.*]] = fir.embox %[[xCast]] typeparams %c10{{.*}} : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.char<1,?>>>
end subroutine

subroutine scalar_char_2()
  character(10), save, target :: x
  character(10), pointer :: p => x
! CHECK-LABEL: fir.global internal @_QFscalar_char_2Ep : !fir.box<!fir.ptr<!fir.char<1,10>>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFscalar_char_2Ex) : !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]] : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.char<1,10>>>
end subroutine

subroutine scalar_derived()
  type t
    real :: x
    integer :: i
  end type
  type(t), save, target :: x
  type(t), pointer :: p => x
! CHECK-LABEL: fir.global internal @_QFscalar_derivedEp : !fir.box<!fir.ptr<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFscalar_derivedEx) : !fir.ref<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]] : (!fir.ref<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>) -> !fir.box<!fir.ptr<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>>
end subroutine

subroutine scalar_null()
  real, pointer :: p => NULL()
! CHECK-LABEL: fir.global internal @_QFscalar_nullEp : !fir.box<!fir.ptr<f32>>
  ! CHECK: %[[zero:.*]] = fir.zero_bits !fir.ptr<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[zero]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<f32>>
end subroutine

! -----------------------------------------------------------------------------
!     Test array initial data target that are simple names
! -----------------------------------------------------------------------------

subroutine array()
  real, save, target :: x(100)
  real, pointer :: p(:) => x
! CHECK-LABEL: fir.global internal @_QFarrayEp : !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFarrayEx) : !fir.ref<!fir.array<100xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape %c100{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]](%[[shape]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
end subroutine

subroutine array_char()
  character(10), save, target :: x(20)
  character(:), pointer :: p(:) => x
! CHECK-LABEL: fir.global internal @_QFarray_charEp : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFarray_charEx) : !fir.ref<!fir.array<20x!fir.char<1,10>>>
  ! CHECK: %[[shape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[xCast:.*]] = fir.convert %[[x]] : (!fir.ref<!fir.array<20x!fir.char<1,10>>>) -> !fir.ptr<!fir.array<?x!fir.char<1,?>>>
  ! CHECK: %[[box:.*]] = fir.embox %[[xCast]](%[[shape]]) typeparams %c10{{.*}} : (!fir.ptr<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
end subroutine

subroutine array_char_2()
  character(10), save, target :: x(20)
  character(10), pointer :: p(:) => x
! CHECK-LABEL: fir.global internal @_QFarray_char_2Ep : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFarray_char_2Ex) : !fir.ref<!fir.array<20x!fir.char<1,10>>>
  ! CHECK: %[[shape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]](%[[shape]]) : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
end subroutine

subroutine array_derived()
  type t
    real :: x
    integer :: i
  end type
  type(t), save, target :: x(100)
  type(t), pointer :: p(:) => x
! CHECK-LABEL: fir.global internal @_QFarray_derivedEp : !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFarray_derivedEx) : !fir.ref<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>
  ! CHECK: %[[shape:.*]] = fir.shape %c100{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]](%[[shape]]) : (!fir.ref<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>>
end subroutine

subroutine array_null()
  real, pointer :: p(:) => NULL()
! CHECK-LABEL: fir.global internal @_QFarray_nullEp : !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: %[[zero:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[shape:.*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[zero]](%[[shape]]) : (!fir.ptr<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
end subroutine

! -----------------------------------------------------------------------------
!     Test scalar initial data target that are data references
! -----------------------------------------------------------------------------

subroutine scalar_ref()
  real, save, target :: x(4:100)
  real, pointer :: p => x(50)
! CHECK-LABEL: fir.global internal @_QFscalar_refEp : !fir.box<!fir.ptr<f32>> {
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFscalar_refEx) : !fir.ref<!fir.array<97xf32>>
  ! CHECK: %[[lb:.*]] = fir.convert %c4 : (index) -> i64
  ! CHECK: %[[idx:.*]] = arith.subi %c50{{.*}}, %[[lb]] : i64
  ! CHECK: %[[elt:.*]] = fir.coordinate_of %[[x]], %[[idx]] : (!fir.ref<!fir.array<97xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[elt]] : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<f32>>
end subroutine

subroutine scalar_char_ref()
  character(20), save, target :: x(100)
  character(10), pointer :: p => x(6)(7:16)
! CHECK-LABEL: fir.global internal @_QFscalar_char_refEp : !fir.box<!fir.ptr<!fir.char<1,10>>>
  ! CHECK: %[[x:.*]] = fir.address_of(@_QFscalar_char_refEx) : !fir.ref<!fir.array<100x!fir.char<1,20>>>
  ! CHECK: %[[idx:.*]] = arith.subi %c6{{.*}}, %c1{{.*}} : i64
  ! CHECK: %[[elt:.*]] = fir.coordinate_of %[[x]], %[[idx]] : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, i64) -> !fir.ref<!fir.char<1,20>>
  ! CHECK: %[[eltCast:.*]] = fir.convert %[[elt:.*]] : (!fir.ref<!fir.char<1,20>>) -> !fir.ref<!fir.array<20x!fir.char<1>>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[eltCast]], %{{.*}} : (!fir.ref<!fir.array<20x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[substring:.*]] = fir.convert %[[coor]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[substringCast:.*]] = fir.convert %[[substring]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ptr<!fir.char<1,10>>
  ! CHECK: %[[box:.*]] = fir.embox %[[substringCast]] : (!fir.ptr<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
  ! CHECK: fir.has_value %[[box]] : !fir.box<!fir.ptr<!fir.char<1,10>>>
end subroutine

! -----------------------------------------------------------------------------
!     Test array initial data target that are data references
! -----------------------------------------------------------------------------


subroutine array_ref()
  real, save, target :: x(4:103, 5:104)
  real, pointer :: p(:) => x(10, 20:100:2)
end subroutine

! CHECK-LABEL: fir.global internal @_QFarray_refEp : !fir.box<!fir.ptr<!fir.array<?xf32>>> {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFarray_refEx) : !fir.ref<!fir.array<100x100xf32>>
! CHECK:         %[[VAL_1:.*]] = arith.constant 4 : index
! CHECK:         %[[VAL_2:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 5 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_7:.*]] = arith.constant 10 : i64
! CHECK:         %[[VAL_8:.*]] = fir.undefined index
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_1]] : index
! CHECK:         %[[VAL_11:.*]] = arith.constant 20 : i64
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
! CHECK:         %[[VAL_13:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:         %[[VAL_15:.*]] = arith.constant 100 : i64
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
! CHECK:         %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_18:.*]] = arith.subi %[[VAL_16]], %[[VAL_12]] : index
! CHECK:         %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_14]] : index
! CHECK:         %[[VAL_20:.*]] = arith.divsi %[[VAL_19]], %[[VAL_14]] : index
! CHECK:         %[[VAL_21:.*]] = arith.cmpi sgt, %[[VAL_20]], %[[VAL_17]] : index
! CHECK:         %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_20]], %[[VAL_17]] : index
! CHECK:         %[[VAL_23:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_24:.*]] = fir.slice %[[VAL_7]], %[[VAL_8]], %[[VAL_8]], %[[VAL_12]], %[[VAL_16]], %[[VAL_14]] : (i64, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_25:.*]] = fir.embox %[[VAL_0]](%[[VAL_23]]) {{\[}}%[[VAL_24]]] : (!fir.ref<!fir.array<100x100xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<41xf32>>
! CHECK:         %[[VAL_26:.*]] = fir.embox %[[VAL_0]](%[[VAL_23]]) {{\[}}%[[VAL_24]]] : (!fir.ref<!fir.array<100x100xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:         fir.has_value %[[VAL_26]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:       }
