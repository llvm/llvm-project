! Test LASTPRIVATE with iteration variable.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck --check-prefix=CHECK-FIR %s

!CHECK-LABEL: func @_QPlastprivate_iv_inc

!CHECK:      %[[I2_MEM:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlastprivate_iv_incEi"}
!CHECK:      %[[I2:.*]]:2 = hlfir.declare %[[I2_MEM]] {uniq_name = "_QFlastprivate_iv_incEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK:      %[[LB:.*]] = arith.constant 4 : i32
!CHECK:      %[[UB:.*]] = arith.constant 10 : i32
!CHECK:      %[[STEP:.*]]  = arith.constant 3 : i32
!CHECK:      omp.wsloop private(@{{.*}} %{{.*}} -> %[[I_MEM:.*]] : !fir.ref<i32>) {
!CHECK-NEXT:   omp.loop_nest (%[[IV:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
!CHECK:          %[[I:.*]]:2 = hlfir.declare %[[I_MEM]] {uniq_name = "_QFlastprivate_iv_incEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:          hlfir.assign %[[IV]] to %[[I]]#1 : i32, !fir.ref<i32>
!CHECK:          %[[UB_2:.*]] = arith.constant 10 : i32
!CHECK:          %[[STEP_2:.*]]  = arith.constant 3 : i32
!CHECK:          %[[V:.*]] = arith.addi %[[IV]], %[[STEP_2]] : i32
!CHECK:          %[[C0:.*]] = arith.constant 0 : i32
!CHECK:          %[[STEP_NEG:.*]] = arith.cmpi slt, %[[STEP_2]], %[[C0]] : i32
!CHECK:          %[[V_LT:.*]] = arith.cmpi slt, %[[V]], %[[UB_2]] : i32
!CHECK:          %[[V_GT:.*]] = arith.cmpi sgt, %[[V]], %[[UB_2]] : i32
!CHECK:          %[[CMP:.*]] = arith.select %[[STEP_NEG]], %[[V_LT]], %[[V_GT]] : i1
!CHECK:          fir.if %[[CMP]] {
!CHECK:            hlfir.assign %[[V]] to %[[I]]#1 : i32, !fir.ref<i32>
!CHECK:            %[[I_VAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
!CHECK:            hlfir.assign %[[I_VAL]] to %[[I2]]#0 : i32, !fir.ref<i32>
!CHECK:          }
!CHECK:          omp.yield
!CHECK:        }
!CHECK:      }
subroutine lastprivate_iv_inc()
  integer :: i

  !$omp do lastprivate(i)
  do i = 4, 10, 3
  end do
  !$omp end do
end subroutine

!CHECK-LABEL: func @_QPlastprivate_iv_dec

!CHECK:      %[[I2_MEM:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlastprivate_iv_decEi"}
!CHECK:      %[[I2:.*]]:2 = hlfir.declare %[[I2_MEM]] {uniq_name = "_QFlastprivate_iv_decEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[LB:.*]] = arith.constant 10 : i32
!CHECK:      %[[UB:.*]] = arith.constant 1 : i32
!CHECK:      %[[STEP:.*]]  = arith.constant -3 : i32
!CHECK:      omp.wsloop private(@{{.*}} %{{.*}} -> %[[I_MEM:.*]] : !fir.ref<i32>) {
!CHECK-NEXT:   omp.loop_nest (%[[IV:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
!CHECK:          %[[I:.*]]:2 = hlfir.declare %[[I_MEM]] {uniq_name = "_QFlastprivate_iv_decEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:          hlfir.assign %[[IV]] to %[[I]]#1 : i32, !fir.ref<i32>
!CHECK:          %[[UB_2:.*]] = arith.constant 1 : i32
!CHECK:          %[[STEP_2:.*]]  = arith.constant -3 : i32
!CHECK:          %[[V:.*]] = arith.addi %[[IV]], %[[STEP_2]] : i32
!CHECK:          %[[C0:.*]] = arith.constant 0 : i32
!CHECK:          %[[STEP_NEG:.*]] = arith.cmpi slt, %[[STEP_2]], %[[C0]] : i32
!CHECK:          %[[V_LT:.*]] = arith.cmpi slt, %[[V]], %[[UB_2]] : i32
!CHECK:          %[[V_GT:.*]] = arith.cmpi sgt, %[[V]], %[[UB_2]] : i32
!CHECK:          %[[CMP:.*]] = arith.select %[[STEP_NEG]], %[[V_LT]], %[[V_GT]] : i1
!CHECK:          fir.if %[[CMP]] {
!CHECK:            hlfir.assign %[[V]] to %[[I]]#1 : i32, !fir.ref<i32>
!CHECK:            %[[I_VAL:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
!CHECK:            hlfir.assign %[[I_VAL]] to %[[I2]]#0 : i32, !fir.ref<i32>
!CHECK:          }
!CHECK:          omp.yield
!CHECK:        }
!CHECK:      }
subroutine lastprivate_iv_dec()
  integer :: i

  !$omp do lastprivate(i)
  do i = 10, 1, -3
  end do
  !$omp end do
end subroutine


!CHECK-LABEL:  @_QPlastprivate_iv_i1
subroutine lastprivate_iv_i1
  integer*1 :: i1
  i1=0
!CHECK:    omp.wsloop private({{.*}})
!CHECK:      omp.loop_nest
!CHECK:        fir.if %{{.*}} {
!CHECK:          hlfir.assign %{{.*}} to %[[IV:.*]]#1 : i32, !fir.ref<i8>
!CHECK:          %[[IV_VAL:.*]] = fir.load %[[IV]]#0 : !fir.ref<i8>
!CHECK:          hlfir.assign %[[IV_VAL]] to %{{.*}}#0 : i8, !fir.ref<i8>
!CHECK:        }


  !$omp do lastprivate(i1)
  do i1=1,8
  enddo
!$omp end do
end subroutine

!CHECK:    omp.wsloop private(@_QFlastprivate_iv_pointerEi_private_box_ptr_i32 %{{.*}}#0 -> %[[PRIVATE_IV:.*]] : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
!CHECK:      omp.loop_nest (%[[LOOP_INDEX:.*]]) : i64 
!CHECK:        %[[PRIVATE_IV_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_IV]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFlastprivate_iv_pointerEi"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!CHECK:        %[[LOOP_INDEX_INCR:.*]] = arith.addi %[[LOOP_INDEX]], %{{.*}} : i64
!CHECK:        fir.if %{{.*}} {
!CHECK:          %[[PRIVATE_IV_BOX:.*]] = fir.load %[[PRIVATE_IV_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK:          %[[PRIVATE_IV_ADDR:.*]] = fir.box_addr %[[PRIVATE_IV_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK:          hlfir.assign %[[LOOP_INDEX_INCR]] to %[[PRIVATE_IV_ADDR]] : i64, !fir.ptr<i32>
!CHECK:        }
!CHECK:        omp.yield
!CHECK:      }
!CHECK:    }

!CHECK-FIR:    omp.wsloop private(@_QFlastprivate_iv_pointerEi_private_box_ptr_i32 %{{.*}} -> %[[PRIVATE_IV:.*]] : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
!CHECK-FIR:      omp.loop_nest (%[[LOOP_INDEX:.*]]) : i64
!CHECK-FIR:        %[[PRIVATE_IV_DECL:.*]] = fir.declare %[[PRIVATE_IV]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFlastprivate_iv_pointerEi"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-FIR:        %[[LOOP_INDEX_INCR:.*]] = arith.addi %[[LOOP_INDEX]], %{{.*}} : i64
!CHECK-FIR:        fir.if %{{.*}} {
!CHECK-FIR:          %[[PRIVATE_IV_BOX:.*]] = fir.load %[[PRIVATE_IV_DECL]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!CHECK-FIR:          %[[PRIVATE_IV_ADDR:.*]] = fir.box_addr %[[PRIVATE_IV_BOX]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
!CHECK-FIR:          %[[LOOP_INDEX_CVT:.*]] = fir.convert %[[LOOP_INDEX_INCR]] : (i64) -> i32
!CHECK-FIR:          fir.store %[[LOOP_INDEX_CVT]] to %[[PRIVATE_IV_ADDR]] : !fir.ptr<i32>
!CHECK-FIR:        }
!CHECK-FIR:        omp.yield
!CHECK-FIR:      }
subroutine lastprivate_iv_pointer()
  integer, pointer :: i

  !$omp do lastprivate(i)
  do i = 1, 20
  end do
  !$omp end do
end subroutine
