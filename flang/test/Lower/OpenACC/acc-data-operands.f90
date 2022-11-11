! This test checks lowering of complex OpenACC data operands.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

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

  !CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_array_section

  !CHECK: %[[ARR:.*]] = fir.alloca !fir.array<100xf32>

  !CHECK: %[[C1:.*]] = fir.convert %c1_i32 : (i32) -> i64
  !CHECK: %[[LB1:.*]] = fir.convert %[[C1]] : (i64) -> index
  !CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  !CHECK: %[[STEP1:.*]] = fir.convert %[[C1_I64]] : (i64) -> index
  !CHECK: %[[C50:.*]] = arith.constant 50 : i32
  !CHECK: %[[C50_I64:.*]] = fir.convert %[[C50]] : (i32) -> i64
  !CHECK: %[[UB1:.*]] = fir.convert %[[C50_I64]] : (i64) -> index
  !CHECK: %[[SHAPE1:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  !CHECK: %[[SLICE1:.*]] = fir.slice %[[LB1]], %[[UB1]], %[[STEP1]] : (index, index, index) -> !fir.slice<1>
  !CHECK: %[[ARR_SECTION1:.*]] = fir.embox %[[ARR]](%[[SHAPE1]]) [%[[SLICE1]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
  !CHECK: %[[MEM1:.*]] = fir.alloca !fir.box<!fir.array<50xf32>>
  !CHECK: fir.store %[[ARR_SECTION1]] to %[[MEM1]] : !fir.ref<!fir.box<!fir.array<50xf32>>>

  !CHECK: %[[C51:.*]] = arith.constant 51 : i32
  !CHECK: %[[C51_I64:.*]] = fir.convert %[[C51]] : (i32) -> i64
  !CHECK: %[[LB2:.*]] = fir.convert %[[C51_I64]] : (i64) -> index
  !CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  !CHECK: %[[STEP2:.*]] = fir.convert %[[C1_I64]] : (i64) -> index
  !CHECK: %[[C100:.*]] = arith.constant 100 : i32
  !CHECK: %[[C100_I64:.*]] = fir.convert %[[C100]] : (i32) -> i64
  !CHECK: %[[UB2:.*]] = fir.convert %[[C100_I64]] : (i64) -> index
  !CHECK: %[[SHAPE2:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  !CHECK: %[[SLICE2:.*]] = fir.slice %[[LB2]], %[[UB2]], %[[STEP2]] : (index, index, index) -> !fir.slice<1>
  !CHECK: %[[ARR_SECTION2:.*]] = fir.embox %[[ARR]](%[[SHAPE2]]) [%[[SLICE2]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
  !CHECK: %[[MEM2:.*]] = fir.alloca !fir.box<!fir.array<50xf32>>
  !CHECK: fir.store %[[ARR_SECTION2]] to %[[MEM2]] : !fir.ref<!fir.box<!fir.array<50xf32>>>

  !CHECK: acc.data copyin(%[[MEM1]] : !fir.ref<!fir.box<!fir.array<50xf32>>>) copyout(%[[MEM2]] : !fir.ref<!fir.box<!fir.array<50xf32>>>)

end subroutine

! Testing array sections of a derived-type component
subroutine acc_operand_array_section_component()

  type(wrapper) :: w

  !$acc data copy(w%data(1:20))
  !$acc end data

  !CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_array_section_component
  !CHECK: %[[W:.*]] = fir.alloca !fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>
  !CHECK: %[[FIELD_INDEX:.*]] = fir.field_index data, !fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>
  !CHECK: %[[DATA_COORD:.*]] = fir.coordinate_of %[[W]], %[[FIELD_INDEX]] : (!fir.ref<!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  !CHECK: %[[C1:.*]] = arith.constant 1 : i32
  !CHECK: %[[C1_I64:.*]] = fir.convert %[[C1]] : (i32) -> i64
  !CHECK: %[[LB:.*]] = fir.convert %3 : (i64) -> index
  !CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
  !CHECK: %[[STEP:.*]] = fir.convert %[[C1_I64]] : (i64) -> index
  !CHECK: %[[C20:.*]] = arith.constant 20 : i32
  !CHECK: %[[C20_I64:.*]] = fir.convert %[[C20]] : (i32) -> i64
  !CHECK: %[[UB:.*]] = fir.convert %[[C20_I64]] : (i64) -> index
  !CHECK: %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
  !CHECK: %[[SLICE:.*]] = fir.slice %[[LB]], %[[UB]], %[[STEP]] : (index, index, index) -> !fir.slice<1>
  !CHECK: %[[ARR_SECTION:.*]] = fir.embox %[[DATA_COORD]](%[[SHAPE]]) [%[[SLICE]]] : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<20xf32>>
  !CHECK: %[[MEM:.*]] = fir.alloca !fir.box<!fir.array<20xf32>> 
  !CHECK: fir.store %[[ARR_SECTION]] to %[[MEM]] : !fir.ref<!fir.box<!fir.array<20xf32>>> 
  !CHECK: acc.data copy(%[[MEM]] : !fir.ref<!fir.box<!fir.array<20xf32>>>)

end subroutine

! Testing derived-type component without section
subroutine acc_operand_derived_type_component()
  type(wrapper) :: w

  !$acc data copy(w%data)
  !$acc end data

  !CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_derived_type_component
  !CHECK: %[[W:.*]] = fir.alloca !fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>
  !CHECK: %[[FIELD_INDEX:.*]] = fir.field_index data, !fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>
  !CHECK: %[[DATA_COORD:.*]] = fir.coordinate_of %[[W]], %[[FIELD_INDEX]] : (!fir.ref<!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  !CHECK: acc.data copy(%[[DATA_COORD]] : !fir.ref<!fir.array<100xf32>>) {

end subroutine

! Testing array of derived-type component without section
subroutine acc_operand_array_derived_type_component()
  type(wrapper) :: w(10)

  !$acc data copy(w(1)%data)
  !$acc end data

  !CHECK-LABEL: func.func @_QMacc_data_operandPacc_operand_array_derived_type_component
  !CHECK: %[[W:.*]] = fir.alloca !fir.array<10x!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>
  !CHECK: %[[IDX:.*]] = arith.subi %{{.*}}, %c1_i64 : i64
  !CHECK: %[[COORD1:.*]] = fir.coordinate_of %[[W]], %[[IDX]] : (!fir.ref<!fir.array<10x!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>>, i64) -> !fir.ref<!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>
  !CHECK: %[[COORD2:.*]] = fir.field_index data, !fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>
  !CHECK: %[[COORD_OF:.*]] = fir.coordinate_of %[[COORD1]], %[[COORD2]] : (!fir.ref<!fir.type<_QMacc_data_operandTwrapper{data:!fir.array<100xf32>}>>, !fir.field) -> !fir.ref<!fir.array<100xf32>>
  !CHECK: acc.data copy(%[[COORD_OF]] : !fir.ref<!fir.array<100xf32>>)

end subroutine

! Testing array sections on allocatable array
subroutine acc_operand_array_section_allocatable()
  real, allocatable :: a(:)

  allocate(a(100))

  !$acc data copyin(a(1:50)) copyout(a(51:100))
  !$acc end data

  !CHECK: %[[ARR_HEAP:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>> {uniq_name = "_QMacc_data_operandFacc_operand_array_section_allocatableEa.addr"}

  !CHECK: %[[LOAD_ARR0:.*]] = fir.load %[[ARR_HEAP]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  !CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
  !CHECK: %[[C1_I64:.*]] = fir.convert %[[C1_I32]] : (i32) -> i64
  !CHECK: %[[LB0:.*]] = fir.convert %[[C1_I64]] : (i64) -> index
  !CHECK: %[[C1_STEP:.*]] = arith.constant 1 : i64
  !CHECK: %[[STEP0:.*]] = fir.convert %[[C1_STEP]] : (i64) -> index
  !CHECK: %[[C50_I32:.*]] = arith.constant 50 : i32
  !CHECK: %[[C50_I64:.*]] = fir.convert %[[C50_I32]] : (i32) -> i64
  !CHECK: %[[UB0:.*]] = fir.convert %[[C50_I64]] : (i64) -> index
  !CHECK: %[[SHAPE_SHIFT0:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
  !CHECK: %[[SLICE0:.*]] = fir.slice %[[LB0]], %[[UB0]], %[[STEP0]] : (index, index, index) -> !fir.slice<1>
  !CHECK: %[[ARR_SECTION0:.*]] = fir.embox %[[LOAD_ARR0]](%[[SHAPE_SHIFT0]]) [%[[SLICE0]]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
  !CHECK: %[[MEM0:.*]] = fir.alloca !fir.box<!fir.array<50xf32>>
  !CHECK: fir.store %[[ARR_SECTION0]] to %[[MEM0]] : !fir.ref<!fir.box<!fir.array<50xf32>>>

  !CHECK: %[[LOAD_ARR1:.*]] = fir.load %[[ARR_HEAP]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
  !CHECK: %[[C51_I32:.*]] = arith.constant 51 : i32
  !CHECK: %[[C51_I64:.*]] = fir.convert %[[C51_I32]] : (i32) -> i64
  !CHECK: %[[LB1:.*]] = fir.convert %[[C51_I64]] : (i64) -> index
  !CHECK: %[[C1_STEP:.*]] = arith.constant 1 : i64
  !CHECK: %[[STEP1:.*]] = fir.convert %[[C1_STEP]] : (i64) -> index
  !CHECK: %[[C100_I32:.*]] = arith.constant 100 : i32
  !CHECK: %[[C100_I64:.*]] = fir.convert %[[C100_I32]] : (i32) -> i64
  !CHECK: %[[UB1:.*]] = fir.convert %[[C100_I64]] : (i64) -> index
  !CHECK: %[[SHAPE_SHIFT1:.*]] = fir.shape_shift %{{.*}}, %{{.*}} : (index, index) -> !fir.shapeshift<1>
  !CHECK: %[[SLICE1:.*]] = fir.slice %[[LB1]], %[[UB1]], %[[STEP1]] : (index, index, index) -> !fir.slice<1>
  !CHECK: %[[ARR_SECTION1:.*]] = fir.embox %[[LOAD_ARR1]](%[[SHAPE_SHIFT1]]) [%[[SLICE1]]] : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
  !CHECK: %[[MEM1:.*]] = fir.alloca !fir.box<!fir.array<50xf32>>
  !CHECK: fir.store %[[ARR_SECTION1]] to %[[MEM1]] : !fir.ref<!fir.box<!fir.array<50xf32>>>

  !CHECK: acc.data copyin(%[[MEM0]] : !fir.ref<!fir.box<!fir.array<50xf32>>>) copyout(%[[MEM1]] : !fir.ref<!fir.box<!fir.array<50xf32>>>)

  deallocate(a)
end subroutine


! Testing array sections on pointer array
subroutine acc_operand_array_section_pointer()
  real, target :: a(100)
  real, pointer :: p(:)

  p => a

  !$acc data copyin(p(1:50))
  !$acc end data

  !CHECK: %[[C100:.*]] = arith.constant 100 : index
  !CHECK: %[[ARR:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "a", fir.target, uniq_name = "_QMacc_data_operandFacc_operand_array_section_pointerEa"}
  !CHECK: %[[PTR:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>> {bindc_name = "p", uniq_name = "_QMacc_data_operandFacc_operand_array_section_pointerEp"}
  !CHECK: %[[SHAPE0:.*]] = fir.shape %[[C100]] : (index) -> !fir.shape<1>
  !CHECK: %[[EMBOX0:.*]] = fir.embox %[[ARR]](%[[SHAPE0]]) : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
  !CHECK: fir.store %[[EMBOX0]] to %[[PTR]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  !CHECK: %[[PTR_LOAD:.*]] = fir.load %[[PTR]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  !CHECK: %[[C0:.*]] = arith.constant 0 : index
  !CHECK: %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[PTR_LOAD]], %[[C0]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
  !CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
  !CHECK: %[[C1_I64:.*]] = fir.convert %[[C1_I32]] : (i32) -> i64
  !CHECK: %[[LB0:.*]] = fir.convert %[[C1_I64]] : (i64) -> index
  !CHECK: %[[C1_STEP:.*]] = arith.constant 1 : i64
  !CHECK: %[[STEP0:.*]] = fir.convert %[[C1_STEP]] : (i64) -> index
  !CHECK: %[[C50_I32:.*]] = arith.constant 50 : i32
  !CHECK: %[[C50_I64:.*]] = fir.convert %[[C50_I32]] : (i32) -> i64
  !CHECK: %[[UB0:.*]] = fir.convert %[[C50_I64]] : (i64) -> index
  !CHECK: %[[SHIFT0:.*]] = fir.shift %[[BOX_DIMS]]#0 : (index) -> !fir.shift<1>
  !CHECK: %[[SLICE0:.*]] = fir.slice %[[LB0]], %[[UB0]], %[[STEP0]] : (index, index, index) -> !fir.slice<1>
  !CHECK: %[[REBOX0:.*]] = fir.rebox %7(%[[SHIFT0]]) [%[[SLICE0]]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<50xf32>>
  !CHECK: %[[MEM0:.*]] = fir.alloca !fir.box<!fir.array<50xf32>>
  !CHECK: fir.store %[[REBOX0]] to %[[MEM0]] : !fir.ref<!fir.box<!fir.array<50xf32>>>
  
  !CHECK: acc.data copyin(%[[MEM0]] : !fir.ref<!fir.box<!fir.array<50xf32>>>) {

end subroutine

end module
