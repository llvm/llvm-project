! Test lowering of pointer initial target
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! -----------------------------------------------------------------------------
!     Test scalar initial data target that are simple names
! -----------------------------------------------------------------------------

subroutine scalar()
  real, save, target :: x
  real, pointer :: p => x
! CHECK-LABEL: fir.global internal @_QFscalarEp : !fir.box<!fir.ptr<f32>>
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFscalarEx) : !fir.ref<f32>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]] {{.*}} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0 : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<f32>>
end subroutine

subroutine scalar_char()
  character(10), save, target :: x
  character(:), pointer :: p => x
! CHECK-LABEL: fir.global internal @_QFscalar_charEp : !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFscalar_charEx) : !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]] typeparams %{{.*}} {{.*}} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.char<1,?>>>
end subroutine

subroutine scalar_char_2()
  character(10), save, target :: x
  character(10), pointer :: p => x
! CHECK-LABEL: fir.global internal @_QFscalar_char_2Ep : !fir.box<!fir.ptr<!fir.char<1,10>>>
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFscalar_char_2Ex) : !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]] typeparams %{{.*}} {{.*}} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.char<1,10>>>
end subroutine

subroutine scalar_derived()
  type t
    real :: x
    integer :: i
  end type
  type(t), save, target :: x
  type(t), pointer :: p => x
! CHECK-LABEL: fir.global internal @_QFscalar_derivedEp : !fir.box<!fir.ptr<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>>
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFscalar_derivedEx) : !fir.ref<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]] {{.*}} : (!fir.ref<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>) -> (!fir.ref<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>, !fir.ref<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>)
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0 : (!fir.ref<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>) -> !fir.box<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>) -> !fir.box<!fir.ptr<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.type<_QFscalar_derivedTt{x:f32,i:i32}>>>
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
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFarrayEx) : !fir.ref<!fir.array<100xf32>>
  ! CHECK: %[[xShape:.*]] = fir.shape %c100{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]](%[[xShape]]) {{.*}} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
  ! CHECK: %[[shape:.*]] = fir.shape %c100{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0(%[[shape]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xf32>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.array<100xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
end subroutine

subroutine array_char()
  character(10), save, target :: x(20)
  character(:), pointer :: p(:) => x
! CHECK-LABEL: fir.global internal @_QFarray_charEp : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFarray_charEx) : !fir.ref<!fir.array<20x!fir.char<1,10>>>
  ! CHECK: %[[xShape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]](%[[xShape]]) typeparams %{{.*}} {{.*}} : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.ref<!fir.array<20x!fir.char<1,10>>>)
  ! CHECK: %[[shape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0(%[[shape]]) : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<20x!fir.char<1,10>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.array<20x!fir.char<1,10>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,?>>>>
end subroutine

subroutine array_char_2()
  character(10), save, target :: x(20)
  character(10), pointer :: p(:) => x
! CHECK-LABEL: fir.global internal @_QFarray_char_2Ep : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFarray_char_2Ex) : !fir.ref<!fir.array<20x!fir.char<1,10>>>
  ! CHECK: %[[xShape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]](%[[xShape]]) typeparams %{{.*}} {{.*}} : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.ref<!fir.array<20x!fir.char<1,10>>>)
  ! CHECK: %[[shape:.*]] = fir.shape %c20{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0(%[[shape]]) : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.array<20x!fir.char<1,10>>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.array<20x!fir.char<1,10>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,10>>>>
end subroutine

subroutine array_derived()
  type t
    real :: x
    integer :: i
  end type
  type(t), save, target :: x(100)
  type(t), pointer :: p(:) => x
! CHECK-LABEL: fir.global internal @_QFarray_derivedEp : !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>>
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFarray_derivedEx) : !fir.ref<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>
  ! CHECK: %[[xShape:.*]] = fir.shape %c100{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]](%[[xShape]]) {{.*}} : (!fir.ref<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>, !fir.ref<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>)
  ! CHECK: %[[shape:.*]] = fir.shape %c100{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[box:.*]] = fir.embox %[[x]]#0(%[[shape]]) : (!fir.ref<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.array<100x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFarray_derivedTt{x:f32,i:i32}>>>>
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
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFscalar_refEx) : !fir.ref<!fir.array<97xf32>>
  ! CHECK: %[[xShape:.*]] = fir.shape_shift %c4{{.*}}, %c97{{.*}} : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]](%[[xShape]]) {{.*}} : (!fir.ref<!fir.array<97xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<97xf32>>, !fir.ref<!fir.array<97xf32>>)
  ! CHECK: %[[elt:.*]] = hlfir.designate %[[x]]#0 (%c50{{.*}})  : (!fir.box<!fir.array<97xf32>>, index) -> !fir.ref<f32>
  ! CHECK: %[[box:.*]] = fir.embox %[[elt]] : (!fir.ref<f32>) -> !fir.box<f32>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<f32>) -> !fir.box<!fir.ptr<f32>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<f32>>
end subroutine

subroutine scalar_char_ref()
  character(20), save, target :: x(100)
  character(10), pointer :: p => x(6)(7:16)
! CHECK-LABEL: fir.global internal @_QFscalar_char_refEp : !fir.box<!fir.ptr<!fir.char<1,10>>>
  ! CHECK: %[[xAddr:.*]] = fir.address_of(@_QFscalar_char_refEx) : !fir.ref<!fir.array<100x!fir.char<1,20>>>
  ! CHECK: %[[xShape:.*]] = fir.shape %c100{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[x:.*]]:2 = hlfir.declare %[[xAddr]](%[[xShape]]) typeparams %c20{{.*}} {{.*}} : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<100x!fir.char<1,20>>>, !fir.ref<!fir.array<100x!fir.char<1,20>>>)
  ! CHECK: %[[elt:.*]] = hlfir.designate %[[x]]#0 (%c6{{.*}}) substr %c7{{.*}}, %c16{{.*}}  typeparams %c10{{.*}} : (!fir.ref<!fir.array<100x!fir.char<1,20>>>, index, index, index, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[box:.*]] = fir.embox %[[elt]] : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
  ! CHECK: %[[rebox:.*]] = fir.rebox %[[box]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
  ! CHECK: fir.has_value %[[rebox]] : !fir.box<!fir.ptr<!fir.char<1,10>>>
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
! CHECK:         %[[VAL_SHAPE:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_X:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_SHAPE]]) {{.*}} : (!fir.ref<!fir.array<100x100xf32>>, !fir.shapeshift<2>) -> (!fir.box<!fir.array<100x100xf32>>, !fir.ref<!fir.array<100x100xf32>>)
! CHECK:         %[[VAL_25:.*]] = hlfir.designate %[[VAL_X]]#0 (%c10{{.*}}, %c20{{.*}}:%c100{{.*}}:%c2{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<100x100xf32>>, index, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<41xf32>>
! CHECK:         %[[VAL_26:.*]] = fir.rebox %[[VAL_25]] : (!fir.box<!fir.array<41xf32>>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:         fir.has_value %[[VAL_26]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:       }
