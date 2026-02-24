! REQUIRES: flang-supports-f128-math
! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -emit-hlfir -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s

! CHECK-LABEL: @_QPadd_test
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<f32>> {{.*}})
subroutine add_test(a,b,c)
  complex :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFadd_testEa"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFadd_testEb"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFadd_testEc"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<f32>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<f32>>
  ! CHECK: %[[VAL_6:.*]] = fir.addc %[[VAL_4]], %[[VAL_5]] {fastmath = #arith.fastmath<contract>} : complex<f32>
  ! CHECK: hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : complex<f32>, !fir.ref<complex<f32>>
  a = b + c
end subroutine add_test

! CHECK-LABEL: @_QPsub_test
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<f32>> {{.*}})
subroutine sub_test(a,b,c)
  complex :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFsub_testEa"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFsub_testEb"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFsub_testEc"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<f32>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<f32>>
  ! CHECK: %[[VAL_6:.*]] = fir.subc %[[VAL_4]], %[[VAL_5]] {fastmath = #arith.fastmath<contract>} : complex<f32>
  ! CHECK: hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : complex<f32>, !fir.ref<complex<f32>>
  a = b - c
end subroutine sub_test

! CHECK-LABEL: @_QPmul_test
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<f32>> {{.*}})
subroutine mul_test(a,b,c)
  complex :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFmul_testEa"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFmul_testEb"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFmul_testEc"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<f32>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<f32>>
  ! CHECK: %[[VAL_6:.*]] = fir.mulc %[[VAL_4]], %[[VAL_5]] {fastmath = #arith.fastmath<contract>} : complex<f32>
  ! CHECK: hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : complex<f32>, !fir.ref<complex<f32>>
  a = b * c
end subroutine mul_test

! CHECK-LABEL: @_QPdiv_test_half
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<f16>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<f16>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<f16>> {{.*}})
subroutine div_test_half(a,b,c)
  complex(kind=2) :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdiv_test_halfEa"} : (!fir.ref<complex<f16>>, !fir.dscope) -> (!fir.ref<complex<f16>>, !fir.ref<complex<f16>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFdiv_test_halfEb"} : (!fir.ref<complex<f16>>, !fir.dscope) -> (!fir.ref<complex<f16>>, !fir.ref<complex<f16>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFdiv_test_halfEc"} : (!fir.ref<complex<f16>>, !fir.dscope) -> (!fir.ref<complex<f16>>, !fir.ref<complex<f16>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<f16>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<f16>>
  ! CHECK: %[[VAL_6:.*]] = complex.div %[[VAL_4]], %[[VAL_5]] fastmath<contract> : complex<f16>
  ! CHECK: hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : complex<f16>, !fir.ref<complex<f16>>
  a = b / c
end subroutine div_test_half

! CHECK-LABEL: @_QPdiv_test_bfloat
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<bf16>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<bf16>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<bf16>> {{.*}})
subroutine div_test_bfloat(a,b,c)
  complex(kind=3) :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdiv_test_bfloatEa"} : (!fir.ref<complex<bf16>>, !fir.dscope) -> (!fir.ref<complex<bf16>>, !fir.ref<complex<bf16>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFdiv_test_bfloatEb"} : (!fir.ref<complex<bf16>>, !fir.dscope) -> (!fir.ref<complex<bf16>>, !fir.ref<complex<bf16>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFdiv_test_bfloatEc"} : (!fir.ref<complex<bf16>>, !fir.dscope) -> (!fir.ref<complex<bf16>>, !fir.ref<complex<bf16>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<bf16>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<bf16>>
  ! CHECK: %[[VAL_6:.*]] = complex.div %[[VAL_4]], %[[VAL_5]] fastmath<contract> : complex<bf16>
  ! CHECK: hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : complex<bf16>, !fir.ref<complex<bf16>>
  a = b / c
end subroutine div_test_bfloat

! CHECK-LABEL: @_QPdiv_test_single
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<f32>> {{.*}})
subroutine div_test_single(a,b,c)
  complex(kind=4) :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdiv_test_singleEa"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFdiv_test_singleEb"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFdiv_test_singleEc"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<f32>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<f32>>
  ! CHECK: %[[VAL_6:.*]] = fir.extract_value %[[VAL_4]], [0 : index] : (complex<f32>) -> f32
  ! CHECK: %[[VAL_7:.*]] = fir.extract_value %[[VAL_4]], [1 : index] : (complex<f32>) -> f32
  ! CHECK: %[[VAL_8:.*]] = fir.extract_value %[[VAL_5]], [0 : index] : (complex<f32>) -> f32
  ! CHECK: %[[VAL_9:.*]] = fir.extract_value %[[VAL_5]], [1 : index] : (complex<f32>) -> f32
  ! CHECK: %[[VAL_10:.*]] = fir.call @__divsc3(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (f32, f32, f32, f32) -> complex<f32>
  ! CHECK: hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : complex<f32>, !fir.ref<complex<f32>>
  a = b / c
end subroutine div_test_single

! CHECK-LABEL: @_QPdiv_test_double
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<f64>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<f64>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<f64>> {{.*}})
subroutine div_test_double(a,b,c)
  complex(kind=8) :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdiv_test_doubleEa"} : (!fir.ref<complex<f64>>, !fir.dscope) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFdiv_test_doubleEb"} : (!fir.ref<complex<f64>>, !fir.dscope) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFdiv_test_doubleEc"} : (!fir.ref<complex<f64>>, !fir.dscope) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<f64>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<f64>>
  ! CHECK: %[[VAL_6:.*]] = fir.extract_value %[[VAL_4]], [0 : index] : (complex<f64>) -> f64
  ! CHECK: %[[VAL_7:.*]] = fir.extract_value %[[VAL_4]], [1 : index] : (complex<f64>) -> f64
  ! CHECK: %[[VAL_8:.*]] = fir.extract_value %[[VAL_5]], [0 : index] : (complex<f64>) -> f64
  ! CHECK: %[[VAL_9:.*]] = fir.extract_value %[[VAL_5]], [1 : index] : (complex<f64>) -> f64
  ! CHECK: %[[VAL_10:.*]] = fir.call @__divdc3(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (f64, f64, f64, f64) -> complex<f64>
  ! CHECK: hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : complex<f64>, !fir.ref<complex<f64>>
  a = b / c
end subroutine div_test_double

! CHECK-LABEL: @_QPdiv_test_extended
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<f80>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<f80>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<f80>> {{.*}})
subroutine div_test_extended(a,b,c)
  complex(kind=10) :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdiv_test_extendedEa"} : (!fir.ref<complex<f80>>, !fir.dscope) -> (!fir.ref<complex<f80>>, !fir.ref<complex<f80>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFdiv_test_extendedEb"} : (!fir.ref<complex<f80>>, !fir.dscope) -> (!fir.ref<complex<f80>>, !fir.ref<complex<f80>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFdiv_test_extendedEc"} : (!fir.ref<complex<f80>>, !fir.dscope) -> (!fir.ref<complex<f80>>, !fir.ref<complex<f80>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<f80>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<f80>>
  ! CHECK: %[[VAL_6:.*]] = fir.extract_value %[[VAL_4]], [0 : index] : (complex<f80>) -> f80
  ! CHECK: %[[VAL_7:.*]] = fir.extract_value %[[VAL_4]], [1 : index] : (complex<f80>) -> f80
  ! CHECK: %[[VAL_8:.*]] = fir.extract_value %[[VAL_5]], [0 : index] : (complex<f80>) -> f80
  ! CHECK: %[[VAL_9:.*]] = fir.extract_value %[[VAL_5]], [1 : index] : (complex<f80>) -> f80
  ! CHECK: %[[VAL_10:.*]] = fir.call @__divxc3(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (f80, f80, f80, f80) -> complex<f80>
  ! CHECK: hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : complex<f80>, !fir.ref<complex<f80>>
  a = b / c
end subroutine div_test_extended

! CHECK-LABEL: @_QPdiv_test_quad
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<complex<f128>> {{.*}}, %[[ARG1:.*]]: !fir.ref<complex<f128>> {{.*}}, %[[ARG2:.*]]: !fir.ref<complex<f128>> {{.*}})
subroutine div_test_quad(a,b,c)
  complex(kind=16) :: a, b, c
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg 1 {uniq_name = "_QFdiv_test_quadEa"} : (!fir.ref<complex<f128>>, !fir.dscope) -> (!fir.ref<complex<f128>>, !fir.ref<complex<f128>>)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg 2 {uniq_name = "_QFdiv_test_quadEb"} : (!fir.ref<complex<f128>>, !fir.dscope) -> (!fir.ref<complex<f128>>, !fir.ref<complex<f128>>)
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %[[VAL_0]] arg 3 {uniq_name = "_QFdiv_test_quadEc"} : (!fir.ref<complex<f128>>, !fir.dscope) -> (!fir.ref<complex<f128>>, !fir.ref<complex<f128>>)
  ! CHECK: %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<complex<f128>>
  ! CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<complex<f128>>
  ! CHECK: %[[VAL_6:.*]] = fir.extract_value %[[VAL_4]], [0 : index] : (complex<f128>) -> f128
  ! CHECK: %[[VAL_7:.*]] = fir.extract_value %[[VAL_4]], [1 : index] : (complex<f128>) -> f128
  ! CHECK: %[[VAL_8:.*]] = fir.extract_value %[[VAL_5]], [0 : index] : (complex<f128>) -> f128
  ! CHECK: %[[VAL_9:.*]] = fir.extract_value %[[VAL_5]], [1 : index] : (complex<f128>) -> f128
  ! CHECK: %[[VAL_10:.*]] = fir.call @__divtc3(%[[VAL_6]], %[[VAL_7]], %[[VAL_8]], %[[VAL_9]]) fastmath<contract> : (f128, f128, f128, f128) -> complex<f128>
  ! CHECK: hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : complex<f128>, !fir.ref<complex<f128>>
  a = b / c
end subroutine div_test_quad
