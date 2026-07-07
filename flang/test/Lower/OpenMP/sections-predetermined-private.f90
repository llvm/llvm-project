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
! CHECK:                 fir.do_loop %[[VAL_12:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:                   %[[VAL_IV:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:                   fir.store %[[VAL_IV]] to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 }
! CHECK:                 %[[LB1:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK:                 %[[UB1:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK:                 %[[ST1:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK:                 %[[C01:.*]] = arith.constant 0 : i32
! CHECK:                 %[[D1:.*]] = arith.subi %[[UB1]], %[[LB1]] overflow<nsw> : i32
! CHECK:                 %[[A1:.*]] = arith.addi %[[D1]], %[[ST1]] overflow<nsw> : i32
! CHECK:                 %[[TR1:.*]] = arith.divsi %[[A1]], %[[ST1]] : i32
! CHECK:                 %[[CMP1:.*]] = arith.cmpi slt, %[[TR1]], %[[C01]] : i32
! CHECK:                 %[[SEL1:.*]] = arith.select %[[CMP1]], %[[C01]], %[[TR1]] : i32
! CHECK:                 %[[M1:.*]] = arith.muli %[[SEL1]], %[[ST1]] overflow<nsw> : i32
! CHECK:                 %[[LAST1:.*]] = arith.addi %[[LB1]], %[[M1]] overflow<nsw> : i32
! CHECK:                 fir.store %[[LAST1]] to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.section {
! CHECK:                 fir.do_loop %[[VAL_26:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
! CHECK:                   %[[VAL_IV2:.*]] = fir.convert %[[VAL_26]] : (index) -> i32
! CHECK:                   fir.store %[[VAL_IV2]] to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 }
! CHECK:                 %[[LB2:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK:                 %[[UB2:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK:                 %[[ST2:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK:                 %[[C02:.*]] = arith.constant 0 : i32
! CHECK:                 %[[D2:.*]] = arith.subi %[[UB2]], %[[LB2]] overflow<nsw> : i32
! CHECK:                 %[[A2:.*]] = arith.addi %[[D2]], %[[ST2]] overflow<nsw> : i32
! CHECK:                 %[[TR2:.*]] = arith.divsi %[[A2]], %[[ST2]] : i32
! CHECK:                 %[[CMP2:.*]] = arith.cmpi slt, %[[TR2]], %[[C02]] : i32
! CHECK:                 %[[SEL2:.*]] = arith.select %[[CMP2]], %[[C02]], %[[TR2]] : i32
! CHECK:                 %[[M2:.*]] = arith.muli %[[SEL2]], %[[ST2]] overflow<nsw> : i32
! CHECK:                 %[[LAST2:.*]] = arith.addi %[[LB2]], %[[M2]] overflow<nsw> : i32
! CHECK:                 fir.store %[[LAST2]] to %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:                 omp.terminator
! CHECK:               }
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
