! Test lowering of complex division according to options

! RUN: bbc -complex-range=full -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,FULL
! RUN: bbc -complex-range=improved -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,IMPRVD
! RUN: bbc -complex-range=basic -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,BASIC
! RUN: %flang_fc1 -complex-range=full -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,FULL
! RUN: %flang_fc1 -complex-range=improved -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,IMPRVD
! RUN: %flang_fc1 -complex-range=basic -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,BASIC


! CHECK-LABEL: @_QPdiv_test_half
! CHECK-SAME: %[[REF_0:.*]]: !fir.ref<complex<f16>> {{.*}}, %[[REF_1:.*]]: !fir.ref<complex<f16>> {{.*}}, %[[REF_2:.*]]: !fir.ref<complex<f16>> {{.*}})
! CHECK: %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[REF_0]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_halfEa"} : (!fir.ref<complex<f16>>, !fir.dscope) -> (!fir.ref<complex<f16>>, !fir.ref<complex<f16>>)
! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[REF_1]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_halfEb"} : (!fir.ref<complex<f16>>, !fir.dscope) -> (!fir.ref<complex<f16>>, !fir.ref<complex<f16>>)
! CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[REF_2]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_halfEc"} : (!fir.ref<complex<f16>>, !fir.dscope) -> (!fir.ref<complex<f16>>, !fir.ref<complex<f16>>)
! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<complex<f16>>
! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<complex<f16>>
! CHECK: %[[VAL_9:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<f16>
! CHECK: hlfir.assign %[[VAL_9]] to %[[VAL_4]]#0 : complex<f16>, !fir.ref<complex<f16>>
! CHECK: return
subroutine div_test_half(a,b,c)
  complex(kind=2) :: a, b, c
  a = b / c
end subroutine div_test_half


! CHECK-LABEL: @_QPdiv_test_bfloat
! CHECK-SAME: %[[REF_0:.*]]: !fir.ref<complex<bf16>> {{.*}}, %[[REF_1:.*]]: !fir.ref<complex<bf16>> {{.*}}, %[[REF_2:.*]]: !fir.ref<complex<bf16>> {{.*}})
! CHECK: %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[REF_0]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_bfloatEa"} : (!fir.ref<complex<bf16>>, !fir.dscope) -> (!fir.ref<complex<bf16>>, !fir.ref<complex<bf16>>)
! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[REF_1]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_bfloatEb"} : (!fir.ref<complex<bf16>>, !fir.dscope) -> (!fir.ref<complex<bf16>>, !fir.ref<complex<bf16>>)
! CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[REF_2]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_bfloatEc"} : (!fir.ref<complex<bf16>>, !fir.dscope) -> (!fir.ref<complex<bf16>>, !fir.ref<complex<bf16>>)
! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<complex<bf16>>
! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<complex<bf16>>
! CHECK: %[[VAL_9:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<bf16>
! CHECK: hlfir.assign %[[VAL_9]] to %[[VAL_4]]#0 : complex<bf16>, !fir.ref<complex<bf16>>
! CHECK: return
subroutine div_test_bfloat(a,b,c)
  complex(kind=3) :: a, b, c
  a = b / c
end subroutine div_test_bfloat


! CHECK-LABEL: @_QPdiv_test_single
! CHECK-SAME: %[[REF_0:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[REF_1:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[REF_2:.*]]: !fir.ref<complex<f32>> {{.*}})
! CHECK: %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[REF_0]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_singleEa"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[REF_1]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_singleEb"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[REF_2]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_singleEc"} : (!fir.ref<complex<f32>>, !fir.dscope) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<complex<f32>>
! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<complex<f32>>

! FULL: %[[VAL_9:.*]] = fir.extract_value %[[VAL_7]], [0 : index] : (complex<f32>) -> f32
! FULL: %[[VAL_10:.*]] = fir.extract_value %[[VAL_7]], [1 : index] : (complex<f32>) -> f32
! FULL: %[[VAL_11:.*]] = fir.extract_value %[[VAL_8]], [0 : index] : (complex<f32>) -> f32
! FULL: %[[VAL_12:.*]] = fir.extract_value %[[VAL_8]], [1 : index] : (complex<f32>) -> f32
! FULL: %[[VAL_RET:.*]] = fir.call @__divsc3(%[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]]) fastmath<contract> : (f32, f32, f32, f32) -> complex<f32>

! IMPRVD: %[[VAL_RET:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<f32>
! BASIC: %[[VAL_RET:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<f32>

! CHECK: hlfir.assign %[[VAL_RET]] to %[[VAL_4]]#0 : complex<f32>, !fir.ref<complex<f32>>
! CHECK: return
subroutine div_test_single(a,b,c)
  complex(kind=4) :: a, b, c
  a = b / c
end subroutine div_test_single


! CHECK-LABEL: @_QPdiv_test_double
! CHECK-SAME: %[[REF_0:.*]]: !fir.ref<complex<f64>> {{.*}}, %[[REF_1:.*]]: !fir.ref<complex<f64>> {{.*}}, %[[REF_2:.*]]: !fir.ref<complex<f64>> {{.*}})
! CHECK: %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[REF_0]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_doubleEa"} : (!fir.ref<complex<f64>>, !fir.dscope) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[REF_1]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_doubleEb"} : (!fir.ref<complex<f64>>, !fir.dscope) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
! CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[REF_2]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_doubleEc"} : (!fir.ref<complex<f64>>, !fir.dscope) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<complex<f64>>
! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<complex<f64>>

! FULL: %[[VAL_9:.*]] = fir.extract_value %[[VAL_7]], [0 : index] : (complex<f64>) -> f64
! FULL: %[[VAL_10:.*]] = fir.extract_value %[[VAL_7]], [1 : index] : (complex<f64>) -> f64
! FULL: %[[VAL_11:.*]] = fir.extract_value %[[VAL_8]], [0 : index] : (complex<f64>) -> f64
! FULL: %[[VAL_12:.*]] = fir.extract_value %[[VAL_8]], [1 : index] : (complex<f64>) -> f64
! FULL: %[[VAL_RET:.*]] = fir.call @__divdc3(%[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]]) fastmath<contract> : (f64, f64, f64, f64) -> complex<f64>

! IMPRVD: %[[VAL_RET:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<f64>
! BASIC: %[[VAL_RET:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<f64>

! CHECK: hlfir.assign %[[VAL_RET]] to %[[VAL_4]]#0 : complex<f64>, !fir.ref<complex<f64>>
! CHECK: return
subroutine div_test_double(a,b,c)
  complex(kind=8) :: a, b, c
  a = b / c
end subroutine div_test_double
