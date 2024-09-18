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
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {uniq_name = "_QMopenacc_boundsFacc_undefined_extentEa"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[DECL_ARG0]]#0, %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK: %[[UB:.*]] = arith.subi %[[DIMS0]]#1, %c1{{.*}} : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%[[UB]] : index) extent(%[[DIMS0]]#1 : index) stride(%[[DIMS0]]#2 : index) startIdx(%c1{{.*}} : index) {strideInBytes = true}
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[DECL_ARG0]]#0 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK: %[[PRESENT:.*]] = acc.present varPtr(%[[ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a"}
! CHECK: acc.kernels dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<?xf32>>)

  subroutine acc_multi_strides(a)
    real, dimension(:,:,:) :: a

    !$acc kernels present(a)
    !$acc end kernels
  end subroutine

! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_multi_strides(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "a"})
! CHECK: %[[DECL_ARG0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMopenacc_boundsFacc_multi_stridesEa"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?x?xf32>>, !fir.box<!fir.array<?x?x?xf32>>)
! CHECK: %[[BOX_DIMS0:.*]]:3 = fir.box_dims %[[DECL_ARG0]]#0, %c0{{.*}} : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
! CHECK: %[[BOUNDS0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%[[BOX_DIMS0]]#1 : index) stride(%[[BOX_DIMS0]]#2 : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! CHECK: %[[STRIDE1:.*]] = arith.muli %[[BOX_DIMS0]]#2, %[[BOX_DIMS0]]#1 : index
! CHECK: %[[BOX_DIMS1:.*]]:3 = fir.box_dims %[[DECL_ARG0]]#0, %c1{{.*}} : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
! CHECK: %[[BOUNDS1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%[[BOX_DIMS1]]#1 : index) stride(%[[STRIDE1]] : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! CHECK: %[[STRIDE2:.*]] = arith.muli %[[STRIDE1]], %[[BOX_DIMS1]]#1 : index
! CHECK: %[[BOX_DIMS2:.*]]:3 = fir.box_dims %[[DECL_ARG0]]#0, %c2{{.*}} : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
! CHECK: %[[BOUNDS2:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%[[BOX_DIMS2]]#1 : index) stride(%[[STRIDE2]] : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[DECL_ARG0]]#0 : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.ref<!fir.array<?x?x?xf32>>
! CHECK: %[[PRESENT:.*]] = acc.present varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?x?x?xf32>>) bounds(%[[BOUNDS0]], %[[BOUNDS1]], %[[BOUNDS2]]) -> !fir.ref<!fir.array<?x?x?xf32>> {name = "a"}
! CHECK: acc.kernels dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<?x?x?xf32>>) {

  subroutine acc_optional_data(a)
    real, pointer, optional :: a(:)
    !$acc data attach(a)
    !$acc end data
  end subroutine
  
! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_optional_data(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "a", fir.optional}) {
! CHECK: %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional, pointer>, uniq_name = "_QMopenacc_boundsFacc_optional_dataEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK: %[[IS_PRESENT:.*]] = fir.is_present %[[ARG0_DECL]]#1 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> i1
! CHECK: %[[RES:.*]]:5 = fir.if %[[IS_PRESENT]] -> (index, index, index, index, index) {
! CHECK:   %[[LOAD:.*]] = fir.load %[[ARG0_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:   fir.result %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : index, index, index, index, index
! CHECK: } else {
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[CM1:.*]] = arith.constant -1 : index
! CHECK:   fir.result %[[C0]], %[[CM1]], %[[C0]], %[[C0]], %[[C0]] : index, index, index, index, index
! CHECK: }
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[RES]]#0 : index) upperbound(%[[RES]]#1 : index) extent(%[[RES]]#2 : index) stride(%[[RES]]#3 : index) startIdx(%[[RES]]#4 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.if %[[IS_PRESENT]] -> (!fir.ptr<!fir.array<?xf32>>) {
! CHECK:   %[[LOAD:.*]] = fir.load %[[ARG0_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:   %[[ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:   fir.result %[[ADDR]] : !fir.ptr<!fir.array<?xf32>>
! CHECK: } else {
! CHECK:   %[[ABSENT:.*]] = fir.absent !fir.ptr<!fir.array<?xf32>>
! CHECK:   fir.result %[[ABSENT]] : !fir.ptr<!fir.array<?xf32>>
! CHECK: }
! CHECK: %[[ATTACH:.*]] = acc.attach varPtr(%[[BOX_ADDR]] : !fir.ptr<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ptr<!fir.array<?xf32>> {name = "a"}
! CHECK: acc.data dataOperands(%[[ATTACH]] : !fir.ptr<!fir.array<?xf32>>)

  subroutine acc_optional_data2(a, n)
    integer :: n
    real, optional :: a(n)
    !$acc data no_create(a)
    !$acc end data
  end subroutine

! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_optional_data2(
! CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "a", fir.optional}, %[[N:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[A]](%{{.*}}) dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMopenacc_boundsFacc_optional_data2Ea"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK: %[[NO_CREATE:.*]] = acc.nocreate varPtr(%[[DECL_A]]#1 : !fir.ref<!fir.array<?xf32>>) bounds(%{{[0-9]+}}) -> !fir.ref<!fir.array<?xf32>> {name = "a"}
! CHECK: acc.data dataOperands(%[[NO_CREATE]] : !fir.ref<!fir.array<?xf32>>) {

  subroutine acc_optional_data3(a, n)
    integer :: n
    real, optional :: a(n)
    !$acc data no_create(a(1:n))
    !$acc end data
  end subroutine

! CHECK-LABEL: func.func @_QMopenacc_boundsPacc_optional_data3(
! CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "a", fir.optional}, %[[N:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[A]](%{{.*}}) dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QMopenacc_boundsFacc_optional_data3Ea"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK: %[[PRES:.*]] = fir.is_present %[[DECL_A]]#1 : (!fir.ref<!fir.array<?xf32>>) -> i1
! CHECK: %[[STRIDE:.*]] = fir.if %[[PRES]] -> (index) {
! CHECK:   %[[DIMS:.*]]:3 = fir.box_dims %[[DECL_A]]#0, %c0{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:   fir.result %[[DIMS]]#2 : index
! CHECK: } else {
! CHECK:   fir.result %c0{{.*}} : index
! CHECK: }
! CHECK: %[[BOUNDS:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%[[STRIDE]] : index) startIdx(%c1 : index) {strideInBytes = true}
! CHECK: %[[NOCREATE:.*]] = acc.nocreate varPtr(%[[DECL_A]]#1 : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xf32>> {name = "a(1:n)"}
! CHECK: acc.data dataOperands(%[[NOCREATE]] : !fir.ref<!fir.array<?xf32>>) {

end module
