! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: @_QPadd_test
subroutine add_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.addc {{.*}}: !fir.complex
  a = b + c
end subroutine add_test

! CHECK-LABEL: @_QPsub_test
subroutine sub_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.subc {{.*}}: !fir.complex
  a = b - c
end subroutine sub_test

! CHECK-LABEL: @_QPmul_test
subroutine mul_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.mulc {{.*}}: !fir.complex
  a = b * c
end subroutine mul_test

! CHECK-LABEL: @_QPdiv_test_half
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<!fir.complex<2>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<2>> {{.*}}, %[[CREF:.*]]: !fir.ref<!fir.complex<2>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<!fir.complex<2>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<!fir.complex<2>>
! CHECK: %[[BVAL_CVT:.*]] = fir.convert %[[BVAL]] : (!fir.complex<2>) -> complex<f16>
! CHECK: %[[CVAL_CVT:.*]] = fir.convert %[[CVAL]] : (!fir.complex<2>) -> complex<f16>
! CHECK: %[[AVAL_CVT:.*]] = complex.div %[[BVAL_CVT]], %[[CVAL_CVT]] : complex<f16>
! CHECK: %[[AVAL:.*]] = fir.convert %[[AVAL_CVT]] : (complex<f16>) -> !fir.complex<2>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<!fir.complex<2>>
subroutine div_test_half(a,b,c)
  complex(kind=2) :: a, b, c
  a = b / c
end subroutine div_test_half

! CHECK-LABEL: @_QPdiv_test_bfloat
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<!fir.complex<3>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<3>> {{.*}}, %[[CREF:.*]]: !fir.ref<!fir.complex<3>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<!fir.complex<3>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<!fir.complex<3>>
! CHECK: %[[BVAL_CVT:.*]] = fir.convert %[[BVAL]] : (!fir.complex<3>) -> complex<bf16>
! CHECK: %[[CVAL_CVT:.*]] = fir.convert %[[CVAL]] : (!fir.complex<3>) -> complex<bf16>
! CHECK: %[[AVAL_CVT:.*]] = complex.div %[[BVAL_CVT]], %[[CVAL_CVT]] : complex<bf16>
! CHECK: %[[AVAL:.*]] = fir.convert %[[AVAL_CVT]] : (complex<bf16>) -> !fir.complex<3>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<!fir.complex<3>>
subroutine div_test_bfloat(a,b,c)
  complex(kind=3) :: a, b, c
  a = b / c
end subroutine div_test_bfloat

! CHECK-LABEL: @_QPdiv_test_single
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}}, %[[CREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<!fir.complex<4>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<!fir.complex<4>>
! CHECK: %[[BREAL:.*]] = fir.extract_value %[[BVAL]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK: %[[BIMAG:.*]] = fir.extract_value %[[BVAL]], [1 : index] : (!fir.complex<4>) -> f32
! CHECK: %[[CREAL:.*]] = fir.extract_value %[[CVAL]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK: %[[CIMAG:.*]] = fir.extract_value %[[CVAL]], [1 : index] : (!fir.complex<4>) -> f32
! CHECK: %[[AVAL:.*]] = fir.call @__divsc3(%[[BREAL]], %[[BIMAG]], %[[CREAL]], %[[CIMAG]]) fastmath<contract> : (f32, f32, f32, f32) -> !fir.complex<4>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<!fir.complex<4>>
subroutine div_test_single(a,b,c)
  complex(kind=4) :: a, b, c
  a = b / c
end subroutine div_test_single

! CHECK-LABEL: @_QPdiv_test_double
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}}, %[[CREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<!fir.complex<8>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<!fir.complex<8>>
! CHECK: %[[BREAL:.*]] = fir.extract_value %[[BVAL]], [0 : index] : (!fir.complex<8>) -> f64
! CHECK: %[[BIMAG:.*]] = fir.extract_value %[[BVAL]], [1 : index] : (!fir.complex<8>) -> f64
! CHECK: %[[CREAL:.*]] = fir.extract_value %[[CVAL]], [0 : index] : (!fir.complex<8>) -> f64
! CHECK: %[[CIMAG:.*]] = fir.extract_value %[[CVAL]], [1 : index] : (!fir.complex<8>) -> f64
! CHECK: %[[AVAL:.*]] = fir.call @__divdc3(%[[BREAL]], %[[BIMAG]], %[[CREAL]], %[[CIMAG]]) fastmath<contract> : (f64, f64, f64, f64) -> !fir.complex<8>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<!fir.complex<8>>
subroutine div_test_double(a,b,c)
  complex(kind=8) :: a, b, c
  a = b / c
end subroutine div_test_double

! CHECK-LABEL: @_QPdiv_test_extended
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<!fir.complex<10>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<10>> {{.*}}, %[[CREF:.*]]: !fir.ref<!fir.complex<10>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<!fir.complex<10>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<!fir.complex<10>>
! CHECK: %[[BREAL:.*]] = fir.extract_value %[[BVAL]], [0 : index] : (!fir.complex<10>) -> f80
! CHECK: %[[BIMAG:.*]] = fir.extract_value %[[BVAL]], [1 : index] : (!fir.complex<10>) -> f80
! CHECK: %[[CREAL:.*]] = fir.extract_value %[[CVAL]], [0 : index] : (!fir.complex<10>) -> f80
! CHECK: %[[CIMAG:.*]] = fir.extract_value %[[CVAL]], [1 : index] : (!fir.complex<10>) -> f80
! CHECK: %[[AVAL:.*]] = fir.call @__divxc3(%[[BREAL]], %[[BIMAG]], %[[CREAL]], %[[CIMAG]]) fastmath<contract> : (f80, f80, f80, f80) -> !fir.complex<10>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<!fir.complex<10>>
subroutine div_test_extended(a,b,c)
  complex(kind=10) :: a, b, c
  a = b / c
end subroutine div_test_extended

! CHECK-LABEL: @_QPdiv_test_quad
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<!fir.complex<16>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<16>> {{.*}}, %[[CREF:.*]]: !fir.ref<!fir.complex<16>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<!fir.complex<16>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<!fir.complex<16>>
! CHECK: %[[BREAL:.*]] = fir.extract_value %[[BVAL]], [0 : index] : (!fir.complex<16>) -> f128
! CHECK: %[[BIMAG:.*]] = fir.extract_value %[[BVAL]], [1 : index] : (!fir.complex<16>) -> f128
! CHECK: %[[CREAL:.*]] = fir.extract_value %[[CVAL]], [0 : index] : (!fir.complex<16>) -> f128
! CHECK: %[[CIMAG:.*]] = fir.extract_value %[[CVAL]], [1 : index] : (!fir.complex<16>) -> f128
! CHECK: %[[AVAL:.*]] = fir.call @__divtc3(%[[BREAL]], %[[BIMAG]], %[[CREAL]], %[[CIMAG]]) fastmath<contract> : (f128, f128, f128, f128) -> !fir.complex<16>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<!fir.complex<16>>
subroutine div_test_quad(a,b,c)
  complex(kind=16) :: a, b, c
  a = b / c
end subroutine div_test_quad
