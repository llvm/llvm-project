! This test checks a few bug fixes in the PRIVATE clause lowering

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: multiple_private_fix
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_private_fixEi"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFmultiple_private_fixEj"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_private_fixEx"}
! CHECK:         omp.parallel {
! CHECK:           %[[PRIV_J:.*]] = fir.alloca i32 {bindc_name = "j", pinned
! CHECK:           %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned
! CHECK:           %[[PRIV_I:.*]] = fir.alloca i32 {{{.*}}, pinned
! CHECK:           %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_4:.*]] : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           omp.wsloop for (%[[VAL_6:.*]]) : i32 = (%[[ONE]]) to (%[[VAL_3]]) inclusive step (%[[VAL_5]]) {
! CHECK:             fir.store %[[VAL_6]] to %[[PRIV_I]] : !fir.ref<i32>
! CHECK:             %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:             %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]] -> index {
! CHECK:               %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:               fir.store %[[VAL_14]] to %[[PRIV_J]] : !fir.ref<i32>
! CHECK:               %[[LOAD:.*]] = fir.load %[[PRIV_I]] : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = fir.load %[[PRIV_J]] : !fir.ref<i32>
! CHECK:               %[[VAL_16:.*]] = arith.addi %[[LOAD]], %[[VAL_15]] : i32
! CHECK:               fir.store %[[VAL_16]] to %[[PRIV_X]] : !fir.ref<i32>
! CHECK:               %[[VAL_17:.*]] = arith.addi %[[VAL_13]], %[[VAL_11]] : index
! CHECK:               fir.result %[[VAL_17]] : index
! CHECK:             }
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_19:.*]] : (index) -> i32
! CHECK:             fir.store %[[VAL_18]] to %[[PRIV_J]] : !fir.ref<i32>
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
