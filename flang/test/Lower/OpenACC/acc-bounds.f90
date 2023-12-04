! This test checks lowering of OpenACC data bounds operation.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module openacc_bounds

type t1
  integer, pointer, dimension(:) :: array_comp
end type

type t2
  integer, dimension(10) :: array_comp
end type

type t3
  integer, allocatable, dimension(:) :: array_comp
end type

contains
  subroutine acc_derived_type_component_pointer_array()
    type(t1) :: d
    !$acc enter data create(d%array_comp)
  end subroutine

! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_derived_type_component_pointer_array() {
! CHECK: %[[D:.*]] = fir.alloca !fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}> {bindc_name = "d", uniq_name = "_QMopenacc_boundsFacc_derived_type_component_pointer_arrayEd"}
! CHECK: %[[DECL_D:.*]]:2 = hlfir.declare %[[D]] {uniq_name = "_QMopenacc_boundsFacc_derived_type_component_pointer_arrayEd"} : (!fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>)
! CHECK: %[[COORD:.*]] = hlfir.designate %[[DECL_D]]#0{"array_comp"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[LOAD:.*]] = fir.load %[[COORD]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[BOX_DIMS0:.*]]:3 = fir.box_dims %[[LOAD]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[BOX_DIMS1:.*]]:3 = fir.box_dims %[[LOAD]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK: %[[UB:.*]] = arith.subi %[[BOX_DIMS1]]#1, %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%[[UB]] : index) extent(%[[BOX_DIMS1]]#1 : index) stride(%[[BOX_DIMS1]]#2 : index) startIdx(%[[BOX_DIMS0]]#0 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ptr<!fir.array<?xi32>>) bounds(%[[BOUND]]) -> !fir.ptr<!fir.array<?xi32>> {name = "d%array_comp", structured = false}
! CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ptr<!fir.array<?xi32>>)
! CHECK: return
! CHECK: }

  subroutine acc_derived_type_component_array()
    type(t2) :: d
    !$acc enter data create(d%array_comp)
  end subroutine

! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_derived_type_component_array()
! CHECK: %[[D:.*]] = fir.alloca !fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}> {bindc_name = "d", uniq_name = "_QMopenacc_boundsFacc_derived_type_component_arrayEd"}
! CHECK: %[[DECL_D:.*]]:2 = hlfir.declare %[[D]] {uniq_name = "_QMopenacc_boundsFacc_derived_type_component_arrayEd"} : (!fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>)
! CHECK: %[[C10:.*]] = arith.constant 10 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
! CHECK: %[[COORD:.*]] = hlfir.designate %[[DECL_D]]#0{"array_comp"} shape %[[SHAPE]] : (!fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[C0]] : index) upperbound(%[[UB]] : index) extent(%[[C10]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[COORD]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "d%array_comp", structured = false}
! CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)
! CHECK: return
! CHECK: }

  subroutine acc_derived_type_component_allocatable_array()
    type(t3) :: d
    !$acc enter data create(d%array_comp)
  end subroutine

! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_derived_type_component_allocatable_array() {
! CHECK: %[[D:.*]] = fir.alloca !fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}> {bindc_name = "d", uniq_name = "_QMopenacc_boundsFacc_derived_type_component_allocatable_arrayEd"}
! CHECK: %[[DECL_D:.*]]:2 = hlfir.declare %[[D]] {uniq_name = "_QMopenacc_boundsFacc_derived_type_component_allocatable_arrayEd"} : (!fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
! CHECK: %[[COORD:.*]] = hlfir.designate %[[DECL_D]]#0{"array_comp"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[LOAD:.*]] = fir.load %[[COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_DIMS0:.*]]:3 = fir.box_dims %[[LOAD]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[BOX_DIMS1:.*]]:3 = fir.box_dims %[[LOAD]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK: %[[UB:.*]] = arith.subi %[[BOX_DIMS1]]#1, %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%[[UB]] : index) extent(%[[BOX_DIMS1]]#1 : index) stride(%[[BOX_DIMS1]]#2 : index) startIdx(%[[BOX_DIMS0]]#0 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>) bounds(%[[BOUND]]) -> !fir.heap<!fir.array<?xi32>> {name = "d%array_comp", structured = false}
! CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xi32>>)
! CHECK: return
! CHECK: }

  subroutine acc_undefined_extent(a)
    real, dimension(1:*) :: a

    !$acc kernels present(a)
    !$acc end kernels
  end subroutine
! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_undefined_extent(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "a"}) {
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {uniq_name = "_QMopenacc_boundsFacc_undefined_extentEa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK: %[[ONE:.*]] = arith.constant 1 : index
! CHECK: %[[ZERO:.*]] = arith.constant 0 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[ZERO]] : index) upperbound(%[[ZERO]] : index) extent(%[[ZERO]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
! CHECK: %[[PRESENT:.*]] = acc.present varPtr(%[[DECL_ARG0]]#1 : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a"}
! CHECK: acc.kernels dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<?xf32>>)

  subroutine acc_multi_strides(a)
    real, dimension(:,:,:) :: a

    !$acc kernels present(a)
    !$acc end kernels
  end subroutine

! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_multi_strides(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "a"})
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QMopenacc_boundsFacc_multi_stridesEa"} : (!fir.box<!fir.array<?x?x?xf32>>) -> (!fir.box<!fir.array<?x?x?xf32>>, !fir.box<!fir.array<?x?x?xf32>>)
! CHECK: %[[BOX_DIMS0:.*]]:3 = fir.box_dims %[[DECL_ARG0]]#1, %c0{{.*}} : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
! CHECK: %[[BOUNDS0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%[[BOX_DIMS0]]#1 : index) stride(%[[BOX_DIMS0]]#2 : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! CHECK: %[[STRIDE1:.*]] = arith.muli %[[BOX_DIMS0]]#2, %[[BOX_DIMS0]]#1 : index
! CHECK: %[[BOX_DIMS1:.*]]:3 = fir.box_dims %[[DECL_ARG0]]#1, %c1{{.*}} : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
! CHECK: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%[[BOX_DIMS1]]#1 : index) stride(%[[STRIDE1]] : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! CHECK: %[[STRIDE2:.*]] = arith.muli %[[STRIDE1]], %[[BOX_DIMS1]]#1 : index
! CHECK: %[[BOX_DIMS2:.*]]:3 = fir.box_dims %[[DECL_ARG0]]#1, %c2{{.*}} : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
! CHECK: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%[[BOX_DIMS2]]#1 : index) stride(%[[STRIDE2]] : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[DECL_ARG0]]#1 : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.ref<!fir.array<?x?x?xf32>>
! CHECK: %[[PRESENT:.*]] = acc.present varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?x?x?xf32>>) bounds(%29, %33, %37) -> !fir.ref<!fir.array<?x?x?xf32>> {name = "a"}
! CHECK: acc.kernels dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<?x?x?xf32>>) {

end module
