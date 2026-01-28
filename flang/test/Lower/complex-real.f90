! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: @_QPsb_complex_bfloat
! CHECK: %[[C_REF:.*]] = fir.alloca complex<bf16> {bindc_name = "c", uniq_name = "_QFsb_complex_bfloatEc"}
! CHECK: %[[C_DECL:.*]]:2 = hlfir.declare %[[C_REF]] {uniq_name = "_QFsb_complex_bfloatEc"} : (!fir.ref<complex<bf16>>) -> (!fir.ref<complex<bf16>>, !fir.ref<complex<bf16>>)
! CHECK: %[[R_REF:.*]] = fir.alloca bf16 {bindc_name = "r", uniq_name = "_QFsb_complex_bfloatEr"}
! CHECK: %[[R_DECL:.*]]:2 = hlfir.declare %[[R_REF]] {uniq_name = "_QFsb_complex_bfloatEr"} : (!fir.ref<bf16>) -> (!fir.ref<bf16>, !fir.ref<bf16>)
! CHECK: %[[R_VAL:.*]] = fir.load %[[R_DECL]]#0 : !fir.ref<bf16>
! CHECK: %[[C_REAL:.*]] = hlfir.designate %[[C_DECL]]#0 real : (!fir.ref<complex<bf16>>) -> !fir.ref<bf16>
! CHECK: hlfir.assign %[[R_VAL]] to %[[C_REAL]] : bf16, !fir.ref<bf16>
subroutine sb_complex_bfloat
  complex(kind=3) :: c
  real(kind=3) :: r
  c%re = r
end subroutine
