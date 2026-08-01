! This test checks lowering of sequential loops in OpenMP parallel.
! The loop indices of these loops should be privatised.

! RUN: bbc -fopenmp -emit-hlfir %s -o - \
! RUN: | FileCheck %s

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - \
! RUN: | FileCheck %s


subroutine sb1
  integer i
  !$omp parallel
  do i=1,10
  end do
  !$omp end parallel

end subroutine

!CHECK-LABEL:  @_QPsb1
!CHECK:    %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsb1Ei"}
!CHECK:    %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ADDR]] {uniq_name = "_QFsb1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.parallel private({{.*}} %[[I_DECL]]#0 -> %[[I_PVT_ADDR:.*]] : {{.*}}) {
!CHECK:      %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_ADDR]] {uniq_name = "_QFsb1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      fir.do_loop %[[I_VAL:.*]] = %[[I_LB:.*]] to %[[I_UB:.*]] step %[[I_ST:.*]] : i32 {
!CHECK:        fir.store %[[I_VAL]] to %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      }
!CHECK:      %[[I_LB_IDX:.*]] = fir.convert %[[I_LB]] : (i32) -> index
!CHECK:      %[[I_UB_IDX:.*]] = fir.convert %[[I_UB]] : (i32) -> index
!CHECK:      %[[I_ST_IDX:.*]] = fir.convert %[[I_ST]] : (i32) -> index
!CHECK:      %[[I_C0:.*]] = arith.constant 0 : index
!CHECK:      %[[I_D:.*]] = arith.subi %[[I_UB_IDX]], %[[I_LB_IDX]] : index
!CHECK:      %[[I_A:.*]] = arith.addi %[[I_D]], %[[I_ST_IDX]] : index
!CHECK:      %[[I_TR:.*]] = arith.divsi %[[I_A]], %[[I_ST_IDX]] : index
!CHECK:      %[[I_CMP:.*]] = arith.cmpi slt, %[[I_TR]], %[[I_C0]] : index
!CHECK:      %[[I_SEL:.*]] = arith.select %[[I_CMP]], %[[I_C0]], %[[I_TR]] : index
!CHECK:      %[[I_M:.*]] = arith.muli %[[I_SEL]], %[[I_ST_IDX]] : index
!CHECK:      %[[I_IDX:.*]] = arith.addi %[[I_LB_IDX]], %[[I_M]] : index
!CHECK:      %[[I_FINAL_VAL:.*]] = fir.convert %[[I_IDX]] : (index) -> i32
!CHECK:      fir.store %[[I_FINAL_VAL]] to %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      omp.terminator
!CHECK:    }
!CHECK:    return
!CHECK:  }

subroutine sb2
  integer i, j, k
  !$omp parallel
  do j=1,10
    if (k .eq. 1) then
      do i=20, 30
      end do
    endif

    do i=40,50
    end do
  end do
  !$omp end parallel
end subroutine
!CHECK-LABEL:  @_QPsb2
!CHECK:    %[[I_ADDR:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsb2Ei"}
!CHECK:    %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ADDR]] {uniq_name = "_QFsb2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[J_ADDR:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsb2Ej"}
!CHECK:    %[[J_DECL:.*]]:2 = hlfir.declare %[[J_ADDR]] {uniq_name = "_QFsb2Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[K_ADDR:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFsb2Ek"}
!CHECK:    %[[K_DECL:.*]]:2 = hlfir.declare %[[K_ADDR]] {uniq_name = "_QFsb2Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.parallel private({{.*}} %[[J_DECL]]#0 -> %[[J_PVT_ADDR:.*]], {{.*}} %[[I_DECL]]#0 -> %[[I_PVT_ADDR:.*]] : {{.*}}) {

!CHECK:      %[[J_PVT_DECL:.*]]:2 = hlfir.declare %[[J_PVT_ADDR]] {uniq_name = "_QFsb2Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:      %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT_ADDR]] {uniq_name = "_QFsb2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:      fir.do_loop %[[J_VAL:.*]] = %[[J_LB:.*]] to %[[J_UB:.*]] step %[[J_ST:.*]] : i32 {
!CHECK:        fir.store %[[J_VAL]] to %[[J_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:        fir.if %{{.*}} {
!CHECK:          fir.do_loop %[[I_VAL:.*]] = %[[I_LB:.*]] to %[[I_UB:.*]] step %[[I_ST:.*]] : i32 {
!CHECK:            fir.store %[[I_VAL]] to %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:          }
!CHECK:          %[[I_LB_IDX:.*]] = fir.convert %[[I_LB]] : (i32) -> index
!CHECK:          %[[I_UB_IDX:.*]] = fir.convert %[[I_UB]] : (i32) -> index
!CHECK:          %[[I_ST_IDX:.*]] = fir.convert %[[I_ST]] : (i32) -> index
!CHECK:          %[[I_C0:.*]] = arith.constant 0 : index
!CHECK:          %[[I_D:.*]] = arith.subi %[[I_UB_IDX]], %[[I_LB_IDX]] : index
!CHECK:          %[[I_A:.*]] = arith.addi %[[I_D]], %[[I_ST_IDX]] : index
!CHECK:          %[[I_TR:.*]] = arith.divsi %[[I_A]], %[[I_ST_IDX]] : index
!CHECK:          %[[I_CMP:.*]] = arith.cmpi slt, %[[I_TR]], %[[I_C0]] : index
!CHECK:          %[[I_SEL:.*]] = arith.select %[[I_CMP]], %[[I_C0]], %[[I_TR]] : index
!CHECK:          %[[I_M:.*]] = arith.muli %[[I_SEL]], %[[I_ST_IDX]] : index
!CHECK:          %[[I_IDX:.*]] = arith.addi %[[I_LB_IDX]], %[[I_M]] : index
!CHECK:          %[[FINAL_I_VAL:.*]] = fir.convert %[[I_IDX]] : (index) -> i32
!CHECK:          fir.store %[[FINAL_I_VAL]] to %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:        }
!CHECK:        fir.do_loop %[[I_VAL2:.*]] = %[[I2_LB:.*]] to %[[I2_UB:.*]] step %[[I2_ST:.*]] : i32 {
!CHECK:          fir.store %[[I_VAL2]] to %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:        }
!CHECK:        %[[I2_LB_IDX:.*]] = fir.convert %[[I2_LB]] : (i32) -> index
!CHECK:        %[[I2_UB_IDX:.*]] = fir.convert %[[I2_UB]] : (i32) -> index
!CHECK:        %[[I2_ST_IDX:.*]] = fir.convert %[[I2_ST]] : (i32) -> index
!CHECK:        %[[I2_C0:.*]] = arith.constant 0 : index
!CHECK:        %[[I2_D:.*]] = arith.subi %[[I2_UB_IDX]], %[[I2_LB_IDX]] : index
!CHECK:        %[[I2_A:.*]] = arith.addi %[[I2_D]], %[[I2_ST_IDX]] : index
!CHECK:        %[[I2_TR:.*]] = arith.divsi %[[I2_A]], %[[I2_ST_IDX]] : index
!CHECK:        %[[I2_CMP:.*]] = arith.cmpi slt, %[[I2_TR]], %[[I2_C0]] : index
!CHECK:        %[[I2_SEL:.*]] = arith.select %[[I2_CMP]], %[[I2_C0]], %[[I2_TR]] : index
!CHECK:        %[[I2_M:.*]] = arith.muli %[[I2_SEL]], %[[I2_ST_IDX]] : index
!CHECK:        %[[I2_IDX:.*]] = arith.addi %[[I2_LB_IDX]], %[[I2_M]] : index
!CHECK:        %[[FINAL_I_VAL2:.*]] = fir.convert %[[I2_IDX]] : (index) -> i32
!CHECK:        fir.store %[[FINAL_I_VAL2]] to %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      }
!CHECK:      %[[J_LB_IDX:.*]] = fir.convert %[[J_LB]] : (i32) -> index
!CHECK:      %[[J_UB_IDX:.*]] = fir.convert %[[J_UB]] : (i32) -> index
!CHECK:      %[[J_ST_IDX:.*]] = fir.convert %[[J_ST]] : (i32) -> index
!CHECK:      %[[J_C0:.*]] = arith.constant 0 : index
!CHECK:      %[[J_D:.*]] = arith.subi %[[J_UB_IDX]], %[[J_LB_IDX]] : index
!CHECK:      %[[J_A:.*]] = arith.addi %[[J_D]], %[[J_ST_IDX]] : index
!CHECK:      %[[J_TR:.*]] = arith.divsi %[[J_A]], %[[J_ST_IDX]] : index
!CHECK:      %[[J_CMP:.*]] = arith.cmpi slt, %[[J_TR]], %[[J_C0]] : index
!CHECK:      %[[J_SEL:.*]] = arith.select %[[J_CMP]], %[[J_C0]], %[[J_TR]] : index
!CHECK:      %[[J_M:.*]] = arith.muli %[[J_SEL]], %[[J_ST_IDX]] : index
!CHECK:      %[[J_IDX:.*]] = arith.addi %[[J_LB_IDX]], %[[J_M]] : index
!CHECK:      %[[FINAL_J_VAL:.*]] = fir.convert %[[J_IDX]] : (index) -> i32
!CHECK:      fir.store %[[FINAL_J_VAL]] to %[[J_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK:      omp.terminator
!CHECK:    }
!CHECK:    return
!CHECK:  }
