! Test different ways of passing the parent component of an extended
! derived-type to a subroutine or the runtime.

! RUN: bbc --use-desc-for-alloc=false -emit-fir %s -o - | FileCheck %s

program parent_comp
  type p
    integer :: a
  end type

  type, extends(p) :: c
    integer :: b
  end type

  type z
    integer :: k
    type(c) :: c
  end type

  type(c) :: t(2) = [ c(11, 21), c(12, 22) ]
  call init_with_slice()
  call init_no_slice()
  call init_allocatable()
  call init_scalar()
  call init_assumed(t)
contains

  subroutine print_scalar(a)
    type(p), intent(in) :: a
    print*, a
  end subroutine
  ! CHECK-LABEL: func.func @_QFPprint_scalar(%{{.*}}: !fir.ref<!fir.type<_QFTp{a:i32}>> {fir.bindc_name = "a"})

  subroutine print_p(a)
    type(p), intent(in) :: a(2)
    print*, a
  end subroutine
  ! CHECK-LABEL: func.func @_QFPprint_p(%{{.*}}: !fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>> {fir.bindc_name = "a"})

  subroutine init_with_slice()
    type(c) :: y(2) = [ c(11, 21), c(12, 22) ]
    call print_p(y(:)%p)
    print*,y(:)%p
  end subroutine
  ! CHECK-LABEL: func.func @_QFPinit_with_slice()
  ! CHECK: %[[Y:.*]] = fir.address_of(@_QFFinit_with_sliceEy) : !fir.ref<!fir.array<2x!fir.type<_QFTc{a:i32,b:i32}>>>
  ! CHECK: %[[C2:.*]] = arith.constant 2 : index
  ! CHECK: %[[C1:.*]] = arith.constant 1 : index
  ! CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  ! CHECK: %[[STRIDE:.*]] = fir.convert %[[C1_I64]] : (i64) -> index
  ! CHECK: %[[ADD:.*]] = arith.addi %[[C1]], %[[C2]] : index
  ! CHECK: %[[UB:.*]] = arith.subi %[[ADD]], %[[C1]] : index
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
  ! CHECK: %[[SLICE:.*]] = fir.slice %[[C1]], %[[UB]], %[[STRIDE]] : (index, index, index) -> !fir.slice<1>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[Y]](%[[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<2x!fir.type<_QFTc{a:i32,b:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> !fir.box<none>
  ! CHECK: %[[IS_CONTIGOUS:.*]] = fir.call @_FortranAIsContiguous(%[[BOX_NONE]]) {{.*}}: (!fir.box<none>) -> i1
  ! CHECK: %[[TEMP:.*]] = fir.if %[[IS_CONTIGOUS]] -> (!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) {
  ! CHECK: } else {
  ! CHECK: fir.call @_FortranAAssign
  ! CHECK: %[[TEMP_CAST:.*]] = fir.convert %[[TEMP]] : (!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> !fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: fir.call @_QFPprint_p(%[[TEMP_CAST]]) {{.*}}: (!fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> ()

  ! CHECK-LABEL: %{{.*}} = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[C1:.*]] = arith.constant 1 : index
  ! CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  ! CHECK: %[[STRIDE:.*]] = fir.convert %[[C1_I64]] : (i64) -> index
  ! CHECK: %[[ADD:.*]] = arith.addi %[[C1]], %[[C2]] : index
  ! CHECK: %[[UB:.*]] = arith.subi %[[ADD]], %[[C1]] : index
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
  ! CHECK: %[[SLICE:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.slice<1>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[Y]](%[[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<2x!fir.type<_QFTc{a:i32,b:i32}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> !fir.box<none>
  ! CHECK: %{{.*}} = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1

  subroutine init_no_slice()
    type(c) :: y(2) = [ c(11, 21), c(12, 22) ]
    call print_p(y%p)
    print*,y%p
  end subroutine
  ! CHECK-LABEL: func.func @_QFPinit_no_slice()
  ! CHECK: %[[Y:.*]] = fir.address_of(@_QFFinit_no_sliceEy) : !fir.ref<!fir.array<2x!fir.type<_QFTc{a:i32,b:i32}>>>
  ! CHECK: %[[C2:.*]] = arith.constant 2 : index
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[Y]](%[[SHAPE]]) : (!fir.ref<!fir.array<2x!fir.type<_QFTc{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> !fir.box<none>
  ! CHECK: %[[IS_CONTIGOUS:.*]] = fir.call @_FortranAIsContiguous(%[[BOX_NONE]]) {{.*}}: (!fir.box<none>) -> i1
  ! CHECK: %[[TEMP:.*]] = fir.if %[[IS_CONTIGOUS]] -> (!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) {
  ! CHECK: } else {
  ! CHECK: fir.call @_FortranAAssign
  ! CHECK: %[[TEMP_CAST:.*]] = fir.convert %[[TEMP]] : (!fir.heap<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> !fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: fir.call @_QFPprint_p(%[[TEMP_CAST]]) {{.*}}: (!fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> ()

  ! CHECK-LABEL: %{{.*}} = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[Y]](%[[SHAPE]]) : (!fir.ref<!fir.array<2x!fir.type<_QFTc{a:i32,b:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> !fir.box<none>
  ! CHECK: %{{.*}} = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1

  subroutine init_allocatable()
    type(c), allocatable :: y(:)
    allocate(y(2))
    y(1) = c(11, 21)
    y(2) = c(12, 22)
    call print_p(y%p)
    print*,y%p
  end subroutine

  ! CHECK-LABEL: func.func @_QFPinit_allocatable()
  ! CHECK: %[[ALLOC:.*]] = fir.alloca !fir.heap<!fir.array<?x!fir.type<_QFTc{a:i32,b:i32}>>> {uniq_name = "_QFFinit_allocatableEy.addr"}
  ! CHECK: %[[LB0:.*]] = fir.alloca index {uniq_name = "_QFFinit_allocatableEy.lb0"}
  ! CHECK: %[[EXT0:.*]] = fir.alloca index {uniq_name = "_QFFinit_allocatableEy.ext0"}
  ! CHECK-COUNT-6: %{{.*}} = fir.field_index a, !fir.type<_QFTc{a:i32,b:i32}>
  ! CHECK: %[[LOAD_LB0:.*]] = fir.load %[[LB0]] : !fir.ref<index>
  ! CHECK: %[[LOAD_EXT0:.*]] = fir.load %[[EXT0]] : !fir.ref<index>
  ! CHECK: %[[MEM:.*]] = fir.load %[[ALLOC]] : !fir.ref<!fir.heap<!fir.array<?x!fir.type<_QFTc{a:i32,b:i32}>>>>
  ! CHECK: %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[LOAD_LB0]], %[[LOAD_EXT0]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[MEM]](%[[SHAPE_SHIFT]]) : (!fir.heap<!fir.array<?x!fir.type<_QFTc{a:i32,b:i32}>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>) -> !fir.box<none> 
  ! CHECK: %[[IS_CONTIGOUS:.*]] = fir.call @_FortranAIsContiguous(%[[BOX_NONE]]) {{.*}}: (!fir.box<none>) -> i1
  ! CHECK: %[[TEMP:.*]] = fir.if %[[IS_CONTIGOUS]] -> (!fir.heap<!fir.array<?x!fir.type<_QFTp{a:i32}>>>) {
  ! CHECK: } else {
  ! CHECK: fir.call @_FortranAAssign
  ! CHECK: %[[TEMP_CAST:.*]] = fir.convert %[[TEMP]] : (!fir.heap<!fir.array<?x!fir.type<_QFTp{a:i32}>>>) -> !fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: fir.call @_QFPprint_p(%[[TEMP_CAST]]) {{.*}}: (!fir.ref<!fir.array<2x!fir.type<_QFTp{a:i32}>>>) -> ()

  ! CHECK-LABEL: %{{.*}} = fir.call @_FortranAioBeginExternalListOutput(%{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[LOAD_LB0:.*]] = fir.load %[[LB0]] : !fir.ref<index>
  ! CHECK: %[[LOAD_EXT0:.*]] = fir.load %[[EXT0]] : !fir.ref<index>
  ! CHECK: %[[LOAD_ALLOC:.*]] = fir.load %[[ALLOC]] : !fir.ref<!fir.heap<!fir.array<?x!fir.type<_QFTc{a:i32,b:i32}>>>>
  ! CHECK: %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[LOAD_LB0]], %[[LOAD_EXT0]] : (index, index) -> !fir.shapeshift<1>
  ! CHECK: %[[BOX:.*]] = fir.embox %[[LOAD_ALLOC]](%[[SHAPE_SHIFT]]) : (!fir.heap<!fir.array<?x!fir.type<_QFTc{a:i32,b:i32}>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>) -> !fir.box<none>
  ! CHECK: %{{.*}} = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1

  subroutine init_scalar()
    type(c) :: s = c(11, 21)
    call print_scalar(s%p)
    print*,s%p
  end subroutine

  ! CHECK-LABEL: func.func @_QFPinit_scalar()
  ! CHECK: %[[S:.*]] = fir.address_of(@_QFFinit_scalarEs) : !fir.ref<!fir.type<_QFTc{a:i32,b:i32}>>
  ! CHECK: %[[CAST:.*]] = fir.convert %[[S]] : (!fir.ref<!fir.type<_QFTc{a:i32,b:i32}>>) -> !fir.ref<!fir.type<_QFTp{a:i32}>>
  ! CHECK: fir.call @_QFPprint_scalar(%[[CAST]]) {{.*}}: (!fir.ref<!fir.type<_QFTp{a:i32}>>) -> ()

  ! CHECK: %[[BOX:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.type<_QFTc{a:i32,b:i32}>>) -> !fir.box<!fir.type<_QFTp{a:i32}>>
  ! CHECK: %[[BOX_NONE:.*]] = fir.convert %[[BOX]] : (!fir.box<!fir.type<_QFTp{a:i32}>>) -> !fir.box<none>
  ! CHECK: %{{.*}} = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[BOX_NONE]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1

  subroutine init_assumed(y)
    type(c) :: y(:)
    call print_p(y%p)
    print*,y%p
  end subroutine

  ! CHECK-LABEL: func.func @_QFPinit_assumed(
  ! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFTc{a:i32,b:i32}>>
  ! CHECK: %[[BOX:.*]] = fir.rebox %[[ARG0]] : (!fir.box<!fir.array<?x!fir.type<_QFTc{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>

  ! CHECK: %[[REBOX:.*]] = fir.rebox %[[ARG0]]  : (!fir.box<!fir.array<?x!fir.type<_QFTc{a:i32,b:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>
  ! CHECK: %[[REBOX_CAST:.*]] = fir.convert %[[REBOX]] : (!fir.box<!fir.array<?x!fir.type<_QFTp{a:i32}>>>) -> !fir.box<none>
  ! CHECK: %{{.*}} = fir.call @_FortranAioOutputDescriptor(%{{.*}}, %[[REBOX_CAST]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1

  subroutine init_existing_field()
    type(z) :: y(2)
    call print_p(y%c%p)
  end subroutine

  ! CHECK-LABEL: func.func @_QFPinit_existing_field
  ! CHECK: %[[C2:.*]] = arith.constant 2 : index
  ! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<2x!fir.type<_QFTz{k:i32,c:!fir.type<_QFTc{a:i32,b:i32}>}>> {bindc_name = "y", uniq_name = "_QFFinit_existing_fieldEy"}
  ! CHECK: %[[FIELD_C:.*]] = fir.field_index c, !fir.type<_QFTz{k:i32,c:!fir.type<_QFTc{a:i32,b:i32}>}>
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
  ! CHECK: %[[C1:.*]] = arith.constant 1 : index
  ! CHECK: %[[SLICE:.*]] = fir.slice %[[C1]], %[[C2]], %[[C1]] path %[[FIELD_C]] : (index, index, index, !fir.field) -> !fir.slice<1>
  ! CHECK: %{{.*}} = fir.embox %[[ALLOCA]](%[[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<2x!fir.type<_QFTz{k:i32,c:!fir.type<_QFTc{a:i32,b:i32}>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<2x!fir.type<_QFTp{a:i32}>>>

  subroutine parent_comp_lhs()
    type(c) :: a
    type(p) :: b

    a%p = B
  end subroutine

! CHECK-LABEL: func.func @_QFPparent_comp_lhs()
! CHECK: %[[BOX:.*]] = fir.alloca !fir.box<!fir.type<_QFTp{a:i32}>>
! CHECK: %[[A:.*]] = fir.alloca !fir.type<_QFTc{a:i32,b:i32}> {bindc_name = "a", uniq_name = "_QFFparent_comp_lhsEa"}
! CHECK: %[[B:.*]] = fir.alloca !fir.type<_QFTp{a:i32}> {bindc_name = "b", uniq_name = "_QFFparent_comp_lhsEb"}
! CHECK: %[[EMBOX_A:.*]] = fir.embox %[[A]] : (!fir.ref<!fir.type<_QFTc{a:i32,b:i32}>>) -> !fir.box<!fir.type<_QFTp{a:i32}>>
! CHECK: %[[EMBOX_B:.*]] = fir.embox %[[B]] : (!fir.ref<!fir.type<_QFTp{a:i32}>>) -> !fir.box<!fir.type<_QFTp{a:i32}>>
! CHECK: fir.store %[[EMBOX_A]] to %[[BOX]] : !fir.ref<!fir.box<!fir.type<_QFTp{a:i32}>>>
! CHECK: %[[A_NONE:.*]] = fir.convert %[[BOX]] : (!fir.ref<!fir.box<!fir.type<_QFTp{a:i32}>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[B_NONE:.*]] = fir.convert %[[EMBOX_B]] : (!fir.box<!fir.type<_QFTp{a:i32}>>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAAssign(%[[A_NONE]], %[[B_NONE]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none

end
