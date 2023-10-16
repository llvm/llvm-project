! This test checks lowering of OpenACC data bounds operation.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

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
! HLFIR: %[[DECL_D:.*]]:2 = hlfir.declare %[[D]] {uniq_name = "_QMopenacc_boundsFacc_derived_type_component_pointer_arrayEd"} : (!fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>)
! FIR: %[[FIELD:.*]] = fir.field_index array_comp, !fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>
! FIR: %[[COORD:.*]] = fir.coordinate_of %[[D]], %[[FIELD]] : (!fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! HLFIR: %[[COORD:.*]] = hlfir.designate %[[DECL_D]]#0{"array_comp"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMopenacc_boundsTt1{array_comp:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[LOAD:.*]] = fir.load %[[COORD]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK: %[[BOX_DIMS0:.*]]:3 = fir.box_dims %[[LOAD]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[BOX_DIMS1:.*]]:3 = fir.box_dims %[[LOAD]], %c0{{.*}} : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK: %[[UB:.*]] = arith.subi %[[BOX_DIMS1]]#1, %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%[[UB]] : index) stride(%[[BOX_DIMS1]]#2 : index) startIdx(%[[BOX_DIMS0]]#0 : index) {strideInBytes = true}
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
! HLFIR: %[[DECL_D:.*]]:2 = hlfir.declare %[[D]] {uniq_name = "_QMopenacc_boundsFacc_derived_type_component_arrayEd"} : (!fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>)
! FIR:   %[[FIELD:.*]] = fir.field_index array_comp, !fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>
! FIR:   %[[COORD:.*]] = fir.coordinate_of %[[D]], %[[FIELD]] : (!fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>, !fir.field) -> !fir.ref<!fir.array<10xi32>>
! CHECK: %[[C10:.*]] = arith.constant 10 : index
! HLFIR: %[[SHAPE:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
! HLFIR: %[[COORD:.*]] = hlfir.designate %[[DECL_D]]#0{"array_comp"} shape %[[SHAPE]] : (!fir.ref<!fir.type<_QMopenacc_boundsTt2{array_comp:!fir.array<10xi32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
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
! HLFIR: %[[DECL_D:.*]]:2 = hlfir.declare %[[D]] {uniq_name = "_QMopenacc_boundsFacc_derived_type_component_allocatable_arrayEd"} : (!fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
! FIR: %[[FIELD:.*]] = fir.field_index array_comp, !fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
! FIR: %[[COORD:.*]] = fir.coordinate_of %[[D]], %[[FIELD]] : (!fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR: %[[COORD:.*]] = hlfir.designate %[[DECL_D]]#0{"array_comp"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QMopenacc_boundsTt3{array_comp:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[LOAD:.*]] = fir.load %[[COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: %[[BOX_DIMS0:.*]]:3 = fir.box_dims %[[LOAD]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[BOX_DIMS1:.*]]:3 = fir.box_dims %[[LOAD]], %c0{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK: %[[UB:.*]] = arith.subi %[[BOX_DIMS1]]#1, %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%[[UB]] : index) stride(%[[BOX_DIMS1]]#2 : index) startIdx(%[[BOX_DIMS0]]#0 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>) bounds(%[[BOUND]]) -> !fir.heap<!fir.array<?xi32>> {name = "d%array_comp", structured = false}
! CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xi32>>)
! CHECK: return
! CHECK: }

end module
