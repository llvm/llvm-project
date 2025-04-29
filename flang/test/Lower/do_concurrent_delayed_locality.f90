! RUN: %flang_fc1 -emit-hlfir -mmlir --openmp-enable-delayed-privatization-staging=true -o - %s | FileCheck %s

subroutine do_concurrent_with_locality_specs
  implicit none
  integer :: i, local_var, local_init_var

  do concurrent (i=1:10) local(local_var) local_init(local_init_var)
    if (i < 5) then
      local_var = 42
    else 
      local_init_var = 84
    end if
  end do
end subroutine

! CHECK-LABEL:   omp.private {type = firstprivate} @_QFdo_concurrent_with_locality_specsElocal_init_var_firstprivate_i32 : i32 copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_1]] : i32, !fir.ref<i32>
! CHECK:           omp.yield(%[[VAL_1]] : !fir.ref<i32>)
! CHECK:         }
! CHECK:         omp.private {type = private} @_QFdo_concurrent_with_locality_specsElocal_var_private_i32 : i32

! CHECK-LABEL:   func.func @_QPdo_concurrent_with_locality_specs() {
! CHECK:           %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFdo_concurrent_with_locality_specsEi"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFdo_concurrent_with_locality_specsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "local_init_var", uniq_name = "_QFdo_concurrent_with_locality_specsElocal_init_var"}
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] {uniq_name = "_QFdo_concurrent_with_locality_specsElocal_init_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "local_var", uniq_name = "_QFdo_concurrent_with_locality_specsElocal_var"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFdo_concurrent_with_locality_specsElocal_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:           %[[VAL_9:.*]] = arith.constant 10 : i32
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:           fir.do_concurrent {
! CHECK:             %[[VAL_12:.*]] = fir.alloca i32 {bindc_name = "i"}
! CHECK:             %[[VAL_13:.*]]:2 = hlfir.declare %[[VAL_12]] {uniq_name = "_QFdo_concurrent_with_locality_specsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:             fir.do_concurrent.loop (%[[VAL_14:.*]]) = (%[[VAL_8]]) to (%[[VAL_10]]) step (%[[VAL_11]]) private(@_QFdo_concurrent_with_locality_specsElocal_var_private_i32 %[[VAL_6]]#0 -> %[[VAL_15:.*]], @_QFdo_concurrent_with_locality_specsElocal_init_var_firstprivate_i32 %[[VAL_4]]#0 -> %[[VAL_16:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:               %[[VAL_17:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
! CHECK:               fir.store %[[VAL_17]] to %[[VAL_13]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_15]] {uniq_name = "_QFdo_concurrent_with_locality_specsElocal_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:               %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_16]] {uniq_name = "_QFdo_concurrent_with_locality_specsElocal_init_var"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:               %[[VAL_20:.*]] = fir.load %[[VAL_13]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_21:.*]] = arith.constant 5 : i32
! CHECK:               %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_21]] : i32
! CHECK:               fir.if %[[VAL_22]] {
! CHECK:                 %[[VAL_23:.*]] = arith.constant 42 : i32
! CHECK:                 hlfir.assign %[[VAL_23]] to %[[VAL_18]]#0 : i32, !fir.ref<i32>
! CHECK:               } else {
! CHECK:                 %[[VAL_24:.*]] = arith.constant 84 : i32
! CHECK:                 hlfir.assign %[[VAL_24]] to %[[VAL_19]]#0 : i32, !fir.ref<i32>
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK:           return
! CHECK:         }
