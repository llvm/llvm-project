! RUN: %flang_fc1 -fopenmp -emit-hlfir -o - %s | FileCheck %s

!$omp parallel sections
!$omp section
    do i = 1, 2
    end do
!$omp section
    do i = 1, 2
    end do
!$omp end parallel sections
end
! CHECK-LABEL:   func.func @_QQmain() {
! CHECK:           omp.parallel {
! CHECK:             %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFEi"}
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             omp.sections {
! CHECK:               omp.section {
! CHECK:                 fir.do_loop %[[VAL_12:.*]] = %[[LB1:.*]] to %[[UB1:.*]] step %[[ST1:.*]] : i32 {
! CHECK:                   fir.store %[[VAL_12]] to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 }
! CHECK:                 %[[LB1IDX:.*]] = fir.convert %[[LB1]] : (i32) -> index
! CHECK:                 %[[UB1IDX:.*]] = fir.convert %[[UB1]] : (i32) -> index
! CHECK:                 %[[ST1IDX:.*]] = fir.convert %[[ST1]] : (i32) -> index
! CHECK:                 %[[C01:.*]] = arith.constant 0 : index
! CHECK:                 %[[D1:.*]] = arith.subi %[[UB1IDX]], %[[LB1IDX]] : index
! CHECK:                 %[[A1:.*]] = arith.addi %[[D1]], %[[ST1IDX]] : index
! CHECK:                 %[[TR1:.*]] = arith.divsi %[[A1]], %[[ST1IDX]] : index
! CHECK:                 %[[CMP1:.*]] = arith.cmpi slt, %[[TR1]], %[[C01]] : index
! CHECK:                 %[[SEL1:.*]] = arith.select %[[CMP1]], %[[C01]], %[[TR1]] : index
! CHECK:                 %[[M1:.*]] = arith.muli %[[SEL1]], %[[ST1IDX]] : index
! CHECK:                 %[[LAST1IDX:.*]] = arith.addi %[[LB1IDX]], %[[M1]] : index
! CHECK:                 %[[LAST1:.*]] = fir.convert %[[LAST1IDX]] : (index) -> i32
! CHECK:                 fir.store %[[LAST1]] to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.section {
! CHECK:                 fir.do_loop %[[VAL_26:.*]] = %[[LB2:.*]] to %[[UB2:.*]] step %[[ST2:.*]] : i32 {
! CHECK:                   fir.store %[[VAL_26]] to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 }
! CHECK:                 %[[LB2IDX:.*]] = fir.convert %[[LB2]] : (i32) -> index
! CHECK:                 %[[UB2IDX:.*]] = fir.convert %[[UB2]] : (i32) -> index
! CHECK:                 %[[ST2IDX:.*]] = fir.convert %[[ST2]] : (i32) -> index
! CHECK:                 %[[C02:.*]] = arith.constant 0 : index
! CHECK:                 %[[D2:.*]] = arith.subi %[[UB2IDX]], %[[LB2IDX]] : index
! CHECK:                 %[[A2:.*]] = arith.addi %[[D2]], %[[ST2IDX]] : index
! CHECK:                 %[[TR2:.*]] = arith.divsi %[[A2]], %[[ST2IDX]] : index
! CHECK:                 %[[CMP2:.*]] = arith.cmpi slt, %[[TR2]], %[[C02]] : index
! CHECK:                 %[[SEL2:.*]] = arith.select %[[CMP2]], %[[C02]], %[[TR2]] : index
! CHECK:                 %[[M2:.*]] = arith.muli %[[SEL2]], %[[ST2IDX]] : index
! CHECK:                 %[[LAST2IDX:.*]] = arith.addi %[[LB2IDX]], %[[M2]] : index
! CHECK:                 %[[LAST2:.*]] = fir.convert %[[LAST2IDX]] : (index) -> i32
! CHECK:                 fir.store %[[LAST2]] to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
