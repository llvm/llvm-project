! RUN: bbc -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK %s

! CHECK-LABEL: func @_QPtest_srand(
subroutine test_srand()
  integer :: seed = 0
  call srand(seed)
  ! CHECK: %[[VAL_0:.*]] = fir.address_of(@_QFtest_srandEseed) : !fir.ref<i32> 
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_srandEseed"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: fir.call @_QPsrand(%[[VAL_1]]#0) fastmath<contract> : (!fir.ref<i32>) -> () 
  ! CHECK: return
end subroutine test_srand

! CHECK-LABEL: func @_QPtest_irand(
subroutine test_irand()
  integer :: seed = 0
  integer :: result
  result = irand(seed)
  ! CHECK: %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "result", uniq_name = "_QFtest_irandEresult"} 
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_irandEresult"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_2:.*]] = fir.address_of(@_QFtest_irandEseed) : !fir.ref<i32> 
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest_irandEseed"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_4:.*]] = fir.call @_FortranAIrand(%[[VAL_3]]#0) fastmath<contract> : (!fir.ref<i32>) -> i32
  ! CHECK: hlfir.assign %[[VAL_4]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
  ! CHECK: return
end subroutine test_irand

! CHECK-LABEL: func @_QPtest_rand(
subroutine test_rand()
  integer :: seed = 0
  real :: result
  result = rand(seed)
  ! CHECK: %[[VAL_0:.*]] = fir.alloca f32 {bindc_name = "result", uniq_name = "_QFtest_randEresult"} 
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_randEresult"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK: %[[VAL_2:.*]] = fir.address_of(@_QFtest_randEseed) : !fir.ref<i32> 
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFtest_randEseed"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! CHECK: %[[VAL_4:.*]] = fir.call @_FortranARand(%[[VAL_3]]#0, %[[SOURCE:.*]], %[[LINE:.*]]) fastmath<contract> : (!fir.ref<i32>, !fir.ref<i8>, i32) -> f32
  ! CHECK: hlfir.assign %[[VAL_4]] to %[[VAL_1]]#0 : f32, !fir.ref<f32>
  ! CHECK: return
end subroutine test_rand

