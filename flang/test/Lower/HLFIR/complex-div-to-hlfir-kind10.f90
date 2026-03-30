! Test lowering of complex division according to options

! REQUIRES: target=x86_64{{.*}}
! RUN: bbc -complex-range=full -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK
! RUN: bbc -complex-range=improved -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK
! RUN: bbc -complex-range=basic -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK
! RUN: %flang_fc1 -complex-range=full -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK
! RUN: %flang_fc1 -complex-range=improved -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK
! RUN: %flang_fc1 -complex-range=basic -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK

! CHECK-LABEL: @_QPdiv_test_extended
! CHECK-SAME: %[[REF_0:.*]]: !fir.ref<complex<f80>> {{.*}}, %[[REF_1:.*]]: !fir.ref<complex<f80>> {{.*}}, %[[REF_2:.*]]: !fir.ref<complex<f80>> {{.*}})
! CHECK: %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[REF_0]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_extendedEa"} : (!fir.ref<complex<f80>>, !fir.dscope) -> (!fir.ref<complex<f80>>, !fir.ref<complex<f80>>)
! CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[REF_1]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_extendedEb"} : (!fir.ref<complex<f80>>, !fir.dscope) -> (!fir.ref<complex<f80>>, !fir.ref<complex<f80>>)
! CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[REF_2]] dummy_scope %[[VAL_3]] arg {{[0-9]+}} {uniq_name = "_QFdiv_test_extendedEc"} : (!fir.ref<complex<f80>>, !fir.dscope) -> (!fir.ref<complex<f80>>, !fir.ref<complex<f80>>)
! CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<complex<f80>>
! CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<complex<f80>>

! CHECK: %[[VAL_RET:.*]] = complex.div %[[VAL_7]], %[[VAL_8]] fastmath<contract> : complex<f80>

! CHECK: hlfir.assign %[[VAL_RET]] to %[[VAL_4]]#0 : complex<f80>, !fir.ref<complex<f80>>
! CHECK: return
subroutine div_test_extended(a,b,c)
  complex(kind=10) :: a, b, c
  a = b / c
end subroutine div_test_extended
