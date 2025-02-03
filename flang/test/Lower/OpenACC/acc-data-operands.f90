! This test checks lowering of complex OpenACC data operands.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module acc_data_operand

  type wrapper
    real :: data(100)
  end type

contains

! Testing array sections as operands
subroutine acc_operand_array_section()
  real, dimension(100) :: a

  !$acc data copyin(a(1:50)) copyout(a(51:100))
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_array_section
! CHECK: %[[EXT:.*]] = arith.constant 100 : index
! CHECK: %[[ARR:.*]] = fir.alloca !fir.array<100xf32>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARR]]
! CHECK: %[[ONE:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.constant 49 : index
! CHECK: %[[BOUND_1_50:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[EXT]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
! CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND_1_50]]) -> !fir.ref<!fir.array<100xf32>> {name = "a(1:50)"}
! CHECK: %[[ONE:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 50 : index
! CHECK: %[[UB:.*]] = arith.constant 99 : index
! CHECK: %[[BOUND_51_100:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[EXT]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
! CHECK: %[[COPYOUT_CREATE:.*]] = acc.create varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND_51_100]]) -> !fir.ref<!fir.array<100xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a(51:100)"}
! CHECK: acc.data dataOperands(%[[COPYIN]], %[[COPYOUT_CREATE]] : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>) {
! CHECK:   acc.terminator
! CHECK: }
! CHECK: acc.delete accPtr(%[[COPYIN]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND_1_50]]) {dataClause = #acc<data_clause acc_copyin>, name = "a(1:50)"}
! CHECK: acc.copyout accPtr(%[[COPYOUT_CREATE]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND_51_100]]) to varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xf32>>) {name = "a(51:100)"}

! Testing array sections of a derived-type component
subroutine acc_operand_array_section_component()

  type(wrapper) :: w

  !$acc data copy(w%data(1:20))
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_array_section_component() {
! CHECK: %[[W:.*]] = fir.alloca !fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}> {bindc_name = "w", uniq_name = "_QMacc_data_operandFacc_operand_array_section_componentEw"}
! CHECK: %[[DECLW:.*]]:2 = hlfir.declare %[[W]]
! CHECK: %[[EXT:.*]] = arith.constant 100 : index
! CHECK: %[[COORD_DATA:.*]] = hlfir.designate %[[DECLW]]#0{"data"}   shape %{{.*}} : (!fir.ref<!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xf32>>
! CHECK: %[[ONE:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.constant 19 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[EXT]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
! CHECK: %[[COPY_COPYIN:.*]] = acc.copyin varPtr(%[[COORD_DATA]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {dataClause = #acc<data_clause acc_copy>, name = "w%data(1:20)"}
! CHECK: acc.data dataOperands(%[[COPY_COPYIN]] : !fir.ref<!fir.array<100xf32>>) {
! CHECK:   acc.terminator
! CHECK: }
! CHECK: acc.copyout accPtr(%[[COPY_COPYIN]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) to varPtr(%[[COORD_DATA]] : !fir.ref<!fir.array<100xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "w%data(1:20)"}

! Testing derived-type component without section
subroutine acc_operand_derived_type_component()
  type(wrapper) :: w

  !$acc data copy(w%data)
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_derived_type_component() {
! CHECK: %[[W:.*]] = fir.alloca !fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}> {bindc_name = "w", uniq_name = "_QMacc_data_operandFacc_operand_derived_type_componentEw"}
! CHECK: %[[DECLW:.*]]:2 = hlfir.declare %[[W]]
! CHECK: %[[EXT:.*]] = arith.constant 100 : index
! CHECK: %[[COORD_DATA:.*]] = hlfir.designate %[[DECLW]]#0{"data"}   shape %{{.*}} : (!fir.ref<!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xf32>>
! CHECK: %[[ONE:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[EXT]], %[[ONE]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[EXT]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
! CHECK: %[[COPY_COPYIN:.*]] = acc.copyin varPtr(%[[COORD_DATA]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {dataClause = #acc<data_clause acc_copy>, name = "w%data"}
! CHECK: acc.data dataOperands(%[[COPY_COPYIN]] : !fir.ref<!fir.array<100xf32>>) {
! CHECK:   acc.terminator
! CHECK: }
! CHECK: acc.copyout accPtr(%[[COPY_COPYIN]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) to varPtr(%[[COORD_DATA]] : !fir.ref<!fir.array<100xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "w%data"}


! Testing array of derived-type component without section
subroutine acc_operand_array_derived_type_component()
  type(wrapper) :: w(10)

  !$acc data copy(w(1)%data)
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_array_derived_type_component() {
! CHECK: %[[W:.*]] = fir.alloca !fir.array<10x!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>> {bindc_name = "w", uniq_name = "_QMacc_data_operandFacc_operand_array_derived_type_componentEw"}
! CHECK: %[[DECLW:.*]]:2 = hlfir.declare %[[W]]
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[W_1:.*]] = hlfir.designate %[[DECLW]]#0 (%[[C1]])  : (!fir.ref<!fir.array<10x!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>>, index) -> !fir.ref<!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>
! CHECK: %[[EXT:.*]] = arith.constant 100 : index
! CHECK: %[[COORD_W1_DATA:.*]] = hlfir.designate %[[W_1]]{"data"}   shape %{{.*}} : (!fir.ref<!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xf32>>
! CHECK: %[[ONE:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.subi %[[EXT]], %[[ONE]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[EXT]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
! CHECK: %[[COPY_COPYIN:.*]] = acc.copyin varPtr(%[[COORD_W1_DATA]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xf32>> {dataClause = #acc<data_clause acc_copy>, name = "w(1_8)%data"}
! CHECK: acc.data dataOperands(%[[COPY_COPYIN]] : !fir.ref<!fir.array<100xf32>>) {
! CHECK:   acc.terminator
! CHECK: }
! CHECK: acc.copyout accPtr(%[[COPY_COPYIN]] : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) to varPtr(%[[COORD_W1_DATA]] : !fir.ref<!fir.array<100xf32>>) {dataClause = #acc<data_clause acc_copy>, name = "w(1_8)%data"}

! Testing array sections on allocatable array
subroutine acc_operand_array_section_allocatable()
  real, allocatable :: a(:)

  allocate(a(100))

  !$acc data copyin(a(1:50)) copyout(a(51:100))
  !$acc end data

  deallocate(a)
end subroutine

! CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_array_section_allocatable() {
! CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QMacc_data_operandFacc_operand_array_section_allocatableEa"}
! CHECK: %[[DECLA:.*]]:2 = hlfir.declare %[[A]] {fortran_attrs = #fir.var_attrs<allocatable>
! CHECK: %[[LOAD_BOX_A_0:.*]] = fir.load %[[DECLA]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[LOAD_BOX_A_1:.*]] = fir.load %[[DECLA]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_0:.*]]:3 = fir.box_dims %[[LOAD_BOX_A_1]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_1:.*]]:3 = fir.box_dims %[[LOAD_BOX_A_0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.subi %[[C1]], %[[DIMS0_0]]#0 : index
! CHECK: %[[C50:.*]] = arith.constant 50 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C50]], %[[DIMS0_0]]#0 : index
! CHECK: %[[LOAD_BOX_A_2:.*]] = fir.load %[[DECLA]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_2:.*]]:3 = fir.box_dims %[[LOAD_BOX_A_2]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[BOUND_1_50:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[DIMS0_2]]#1 : index) stride(%[[DIMS0_1]]#2 : index) startIdx(%[[DIMS0_0]]#0 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD_BOX_A_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND_1_50]]) -> !fir.heap<!fir.array<?xf32>> {name = "a(1:50)"}
! CHECK: %[[LOAD_BOX_A_0:.*]] = fir.load %[[DECLA]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[LOAD_BOX_A_1:.*]] = fir.load %[[DECLA]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_0:.*]]:3 = fir.box_dims %[[LOAD_BOX_A_1]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_1:.*]]:3 = fir.box_dims %[[LOAD_BOX_A_0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[C51:.*]] = arith.constant 51 : index
! CHECK: %[[LB:.*]] = arith.subi %[[C51]], %[[DIMS0_0]]#0 : index
! CHECK: %[[C100:.*]] = arith.constant 100 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C100]], %[[DIMS0_0]]#0 : index
! CHECK: %[[LOAD_BOX_A_2:.*]] = fir.load %[[DECLA]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_2:.*]]:3 = fir.box_dims %[[LOAD_BOX_A_2]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[BOUND_51_100:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[DIMS0_2]]#1 : index) stride(%[[DIMS0_1]]#2 : index) startIdx(%[[DIMS0_0]]#0 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD_BOX_A_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK: %[[COPYOUT_CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND_51_100]]) -> !fir.heap<!fir.array<?xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a(51:100)"}
! CHECK: acc.data dataOperands(%[[COPYIN]], %[[COPYOUT_CREATE]] : !fir.heap<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>) {
! CHECK:   acc.terminator
! CHECK: }
! CHECK: acc.delete accPtr(%[[COPYIN]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND_1_50]]) {dataClause = #acc<data_clause acc_copyin>, name = "a(1:50)"}
! CHECK: acc.copyout accPtr(%[[COPYOUT_CREATE]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND_51_100]]) to varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>) {name = "a(51:100)"}


! Testing array sections on pointer array
subroutine acc_operand_array_section_pointer()
  real, target :: a(100)
  real, pointer :: p(:)

  p => a

  !$acc data copyin(p(1:50))
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_array_section_pointer() {
! CHECK: %[[P:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "p", uniq_name = "_QMacc_data_operandFacc_operand_array_section_pointerEp"}
! CHECK: %[[DECLP:.*]]:2 = hlfir.declare %[[P]] {fortran_attrs = #fir.var_attrs<pointer>
! CHECK: %[[LOAD_BOX_P_0:.*]] = fir.load %[[DECLP]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[LOAD_BOX_P_1:.*]] = fir.load %[[DECLP]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_0:.*]]:3 = fir.box_dims %[[LOAD_BOX_P_1]], %[[C0:.*]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_1:.*]]:3 = fir.box_dims %[[LOAD_BOX_P_0]], %[[C0]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.subi %[[C1]], %[[DIMS0_0]]#0 : index
! CHECK: %[[C50:.*]] = arith.constant 50 : index
! CHECK: %[[UB:.*]] = arith.subi %[[C50]], %[[DIMS0_0]]#0 : index
! CHECK: %[[LOAD_BOX_P_2:.*]] = fir.load %[[DECLP]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[DIMS0_2:.*]]:3 = fir.box_dims %[[LOAD_BOX_P_2]], %[[C0]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[DIMS0_2]]#1 : index) stride(%[[DIMS0_1]]#2 : index) startIdx(%[[DIMS0_0]]#0 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD_BOX_P_0]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[BOX_ADDR]] : !fir.ptr<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ptr<!fir.array<?xf32>> {name = "p(1:50)"}
! CHECK: acc.data dataOperands(%[[COPYIN]] : !fir.ptr<!fir.array<?xf32>>) {
! CHECK:   acc.terminator
! CHECK: }
! CHECK: acc.delete accPtr(%[[COPYIN]] : !fir.ptr<!fir.array<?xf32>>) bounds(%[[BOUND]]) {dataClause = #acc<data_clause acc_copyin>, name = "p(1:50)"}


end module
