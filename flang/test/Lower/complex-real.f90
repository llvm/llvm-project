! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: @_QPsb_complex_bfloat
! CHECK: %[[C_REF:.*]] = fir.alloca complex<bf16> {bindc_name = "c", uniq_name = "_QFsb_complex_bfloatEc"}
! CHECK: %[[R_REF:.*]] = fir.alloca bf16 {bindc_name = "r", uniq_name = "_QFsb_complex_bfloatEr"}
! CHECK: %[[R_VAL:.*]] = fir.load %[[R_REF]] : !fir.ref<bf16>
! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: %[[CREAL_REF:.*]] = fir.coordinate_of %[[C_REF]], %[[C0]] : (!fir.ref<complex<bf16>>, i32) -> !fir.ref<bf16>
! CHECK: fir.store %[[R_VAL]] to %[[CREAL_REF]] : !fir.ref<bf16>
subroutine sb_complex_bfloat
  complex(kind=3) :: c
  real(kind=3) :: r
  c%re = r
end subroutine
