! This test checks a few bug fixes in the PRIVATE clause lowering

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: multiple_private_fix
! CHECK-SAME:  %[[GAMA:.*]]: !fir.ref<i32> {fir.bindc_name = "gama"}
! CHECK-DAG:         %[[GAMA_DECL:.*]]:2 = hlfir.declare %[[GAMA]] {uniq_name = "_QFmultiple_private_fixEgama"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_private_fixEi"}
! CHECK-DAG:         %[[I_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmultiple_private_fixEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFmultiple_private_fixEj"}
! CHECK-DAG:         %[[J_DECL:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFmultiple_private_fixEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_private_fixEx"}
! CHECK-DAG:         %[[X_DECL:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFmultiple_private_fixEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         omp.parallel {
! CHECK-DAG:           %[[PRIV_I:.*]] = fir.alloca i32 {adapt.valuebyref, pinned}
! CHECK-DAG:           %[[PRIV_I_DECL:.*]]:2 = hlfir.declare %[[PRIV_I]] {uniq_name = "_QFmultiple_private_fixEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:           %[[PRIV_J:.*]] = fir.alloca i32 {bindc_name = "j", pinned, uniq_name = "_QFmultiple_private_fixEj"}
! CHECK-DAG:           %[[PRIV_J_DECL:.*]]:2 = hlfir.declare %[[PRIV_J]] {uniq_name = "_QFmultiple_private_fixEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:           %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned
! CHECK-DAG:           %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFmultiple_private_fixEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[GAMA_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           omp.wsloop for (%[[VAL_6:.*]]) : i32 = (%[[ONE]]) to (%[[VAL_3]]) inclusive step (%[[VAL_5]]) {
! CHECK:             fir.store %[[VAL_6]] to %[[PRIV_I_DECL]]#1 : !fir.ref<i32>
! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:             %[[VAL_9:.*]] = fir.load %[[GAMA_DECL]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:             %[[LB:.*]] = fir.convert %[[VAL_8]] : (index) -> i32
! CHECK:             %[[VAL_12:.*]]:2 = fir.do_loop %[[VAL_13:[^ ]*]] =
! CHECK-SAME:            %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]]
! CHECK-SAME:            iter_args(%[[IV:.*]] = %[[LB]]) -> (index, i32) {
! CHECK:               fir.store %[[IV]] to %[[PRIV_J_DECL]]#1 : !fir.ref<i32>
! CHECK:               %[[LOAD:.*]] = fir.load %[[PRIV_I_DECL]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = fir.load %[[PRIV_J_DECL]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_16:.*]] = arith.addi %[[LOAD]], %[[VAL_15]] : i32
! CHECK:               hlfir.assign %[[VAL_16]] to %[[PRIV_X_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:               %[[VAL_17:.*]] = arith.addi %[[VAL_13]], %[[VAL_11]] : index
! CHECK:               %[[STEPCAST:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:               %[[IVLOAD:.*]] = fir.load %[[PRIV_J_DECL]]#1 : !fir.ref<i32>
! CHECK:               %[[IVINC:.*]] = arith.addi %[[IVLOAD]], %[[STEPCAST]]
! CHECK:               fir.result %[[VAL_17]], %[[IVINC]] : index, i32
! CHECK:             }
! CHECK:             fir.store %[[VAL_12]]#1 to %[[PRIV_J_DECL]]#1 : !fir.ref<i32>
! CHECK:             omp.yield
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
subroutine multiple_private_fix(gama)
        integer :: i, j, x, gama
!$OMP PARALLEL DO PRIVATE(j,x)
        do i = 1, gama
          do j = 1, gama
            x = i + j
          end do
        end do
!$OMP END PARALLEL DO
end subroutine

! CHECK-LABEL: multiple_private_fix2
! CHECK:  %[[X1:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_private_fix2Ex"}
! CHECK:  %[[X1_DECL:.*]]:2 = hlfir.declare %[[X1]] {uniq_name = "_QFmultiple_private_fix2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  omp.parallel  {
! CHECK:    %[[X2:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFmultiple_private_fix2Ex"}
! CHECK:    %[[X2_DECL:.*]]:2 = hlfir.declare %[[X2]] {uniq_name = "_QFmultiple_private_fix2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    omp.parallel  {
! CHECK:      %[[X3:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFmultiple_private_fix2Ex"}
! CHECK:      %[[X3_DECL:.*]]:2 = hlfir.declare %[[X3]] {uniq_name = "_QFmultiple_private_fix2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:      %[[C3:.*]] = arith.constant 1 : i32
! CHECK:      hlfir.assign %[[C3]] to %[[X3_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:      omp.terminator
! CHECK:    }
! CHECK:      %[[C2:.*]] = arith.constant 1 : i32
! CHECK:      hlfir.assign %[[C2]] to %[[X2_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:    omp.terminator
! CHECK:  }
! CHECK:      %[[C1:.*]] = arith.constant 1 : i32
! CHECK:      hlfir.assign %[[C1]] to %[[X1_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:  return
subroutine multiple_private_fix2()
   integer :: x
   !$omp parallel private(x)
   !$omp parallel private(x)
      x = 1
   !$omp end parallel
      x = 1
   !$omp end parallel
      x = 1
end subroutine
