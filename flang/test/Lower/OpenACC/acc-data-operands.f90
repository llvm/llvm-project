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

subroutine acc_operand_array_section2(a)
  real, dimension(100) :: a

  !$acc data copyin(a)
  !$acc end data

end subroutine

end module
