! This test checks a few bug fixes in the PRIVATE clause lowering

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: multiple_private_fix
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_private_fixEi"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFmultiple_private_fixEj"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_private_fixEx"}
! CHECK:         omp.parallel {
! CHECK:           %[[PRIV_J:.*]] = fir.alloca i32 {bindc_name = "j", pinned
! CHECK:           %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned
! CHECK:           %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_4:.*]] : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           omp.wsloop for (%[[IV_I:.*]]) : i32 = (%[[ONE]]) to (%[[VAL_3]]) inclusive step (%[[VAL_5]]) {
! CHECK:             %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:             %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:             %[[LB:.*]] = fir.convert %[[VAL_7]] : (index) -> i32
! CHECK:             %[[VAL_11:.*]]:2 = fir.do_loop %[[VAL_12:[^ ]*]] =
! CHECK-SAME:            %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]]
! CHECK-SAME:            iter_args(%[[IV_J:.*]] = %[[LB]]) -> (index, i32) {
! CHECK:               fir.store %[[IV_J]] to %[[PRIV_J]] : !fir.ref<i32>
! CHECK:               %[[VAL_13:.*]] = fir.load %[[PRIV_J]] : !fir.ref<i32>
! CHECK:               %[[VAL_14:.*]] = arith.addi %[[IV_I]], %[[VAL_13]] : i32
! CHECK:               fir.store %[[VAL_14]] to %[[PRIV_X]] : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_12]], %[[VAL_10]] : index
! CHECK:               %[[STEPCAST:.*]] = fir.convert %[[VAL_10]] : (index) -> i32
! CHECK:               %[[IVLOAD:.*]] = fir.load %[[PRIV_J]] : !fir.ref<i32>
! CHECK:               %[[IVINC:.*]] = arith.addi %[[IVLOAD]], %[[STEPCAST]]
! CHECK:               fir.result %[[VAL_15]], %[[IVINC]] : index, i32
! CHECK:             }
! CHECK:             fir.store %[[VAL_11]]#1 to %[[PRIV_J]] : !fir.ref<i32>
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
! CHECK:  omp.parallel  {
! CHECK:    %[[X2:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFmultiple_private_fix2Ex"}
! CHECK:    omp.parallel  {
! CHECK:      %[[X3:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFmultiple_private_fix2Ex"}
! CHECK:      %[[C3:.*]] = arith.constant 1 : i32
! CHECK:      fir.store %[[C3]] to %[[X3]] : !fir.ref<i32>
! CHECK:      omp.terminator
! CHECK:    }
! CHECK:      %[[C2:.*]] = arith.constant 1 : i32
! CHECK:      fir.store %[[C2]] to %[[X2]] : !fir.ref<i32>
! CHECK:    omp.terminator
! CHECK:  }
! CHECK:      %[[C1:.*]] = arith.constant 1 : i32
! CHECK:      fir.store %[[C1]] to %[[X1]] : !fir.ref<i32>
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
