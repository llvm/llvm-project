! Test lowering of complex division according to options

! REQUIRES: flang-supports-f128-math
! RUN: bbc -complex-range=full -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,FULL
! RUN: bbc -complex-range=improved -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,IMPRVD
! RUN: bbc -complex-range=basic -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,BASIC
! RUN: %flang_fc1 -complex-range=full -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,FULL
! RUN: %flang_fc1 -complex-range=improved -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,IMPRVD
! RUN: %flang_fc1 -complex-range=basic -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,BASIC


! CHECK-LABEL: @_QPdiv_test_quad
! CHECK-SAME: %[[REF_0:.*]]: !fir.ref<complex<f128>> {{.*}}, %[[REF_1:.*]]: !fir.ref<complex<f128>> {{.*}}, %[[REF_2:.*]]: !fir.ref<complex<f128>> {{.*}})
! CHECK: %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[REF_0]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_quadEa"} : (!fir.ref<complex<f128>>, !fir.dscope) -> (!fir.ref<complex<f128>>, !fir.ref<complex<f128>>)
! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[REF_1]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_quadEb"} : (!fir.ref<complex<f128>>, !fir.dscope) -> (!fir.ref<complex<f128>>, !fir.ref<complex<f128>>)
! CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[REF_2]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_quadEc"} : (!fir.ref<complex<f128>>, !fir.dscope) -> (!fir.ref<complex<f128>>, !fir.ref<complex<f128>>)
! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<complex<f128>>
! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<complex<f128>>

! FULL: %[[VAL_9:.*]] = fir.extract_value %[[VAL_7]], [0 : index] : (complex<f128>) -> f128
! FULL: %[[VAL_10:.*]] = fir.extract_value %[[VAL_7]], [1 : index] : (complex<f128>) -> f128
! FULL: %[[VAL_11:.*]] = fir.extract_value %[[VAL_8]], [0 : index] : (complex<f128>) -> f128
! FULL: %[[VAL_12:.*]] = fir.extract_value %[[VAL_8]], [1 : index] : (complex<f128>) -> f128
! FULL: %[[VAL_RET:.*]] = fir.call @__divtc3(%[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %[[VAL_12]]) fastmath<contract> : (f128, f128, f128, f128) -> complex<f128>

! IMPRVD: %[[VAL_RET:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<f128>
! BASIC: %[[VAL_RET:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<f128>

! CHECK: hlfir.assign %[[VAL_RET]] to %[[VAL_4]]#0 : complex<f128>, !fir.ref<complex<f128>>
! CHECK: return
subroutine div_test_quad(a,b,c)
  complex(kind=16) :: a, b, c
  a = b / c
end subroutine div_test_quad
