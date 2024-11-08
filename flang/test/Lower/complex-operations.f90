! RUN: bbc -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: @_QPadd_test
subroutine add_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.addc {{.*}}: complex
  a = b + c
end subroutine add_test

! CHECK-LABEL: @_QPsub_test
subroutine sub_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.subc {{.*}}: complex
  a = b - c
end subroutine sub_test

! CHECK-LABEL: @_QPmul_test
subroutine mul_test(a,b,c)
  complex :: a, b, c
  ! CHECK-NOT: fir.extract_value
  ! CHECK-NOT: fir.insert_value
  ! CHECK: fir.mulc {{.*}}: complex
  a = b * c
end subroutine mul_test

! CHECK-LABEL: @_QPdiv_test_half
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<complex<f16>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f16>> {{.*}}, %[[CREF:.*]]: !fir.ref<complex<f16>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<complex<f16>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<complex<f16>>
! CHECK: %[[AVAL:.*]] = complex.div %[[BVAL]], %[[CVAL]] fastmath<contract> : complex<f16>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<complex<f16>>
subroutine div_test_half(a,b,c)
  complex(kind=2) :: a, b, c
  a = b / c
end subroutine div_test_half

! CHECK-LABEL: @_QPdiv_test_bfloat
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<complex<bf16>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<bf16>> {{.*}}, %[[CREF:.*]]: !fir.ref<complex<bf16>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<complex<bf16>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<complex<bf16>>
! CHECK: %[[AVAL:.*]] = complex.div %[[BVAL]], %[[CVAL]] fastmath<contract> : complex<bf16>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<complex<bf16>>
subroutine div_test_bfloat(a,b,c)
  complex(kind=3) :: a, b, c
  a = b / c
end subroutine div_test_bfloat

! CHECK-LABEL: @_QPdiv_test_single
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[CREF:.*]]: !fir.ref<complex<f32>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<complex<f32>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<complex<f32>>
! CHECK: %[[BREAL:.*]] = fir.extract_value %[[BVAL]], [0 : index] : (complex<f32>) -> f32
! CHECK: %[[BIMAG:.*]] = fir.extract_value %[[BVAL]], [1 : index] : (complex<f32>) -> f32
! CHECK: %[[CREAL:.*]] = fir.extract_value %[[CVAL]], [0 : index] : (complex<f32>) -> f32
! CHECK: %[[CIMAG:.*]] = fir.extract_value %[[CVAL]], [1 : index] : (complex<f32>) -> f32
! CHECK: %[[AVAL:.*]] = fir.call @__divsc3(%[[BREAL]], %[[BIMAG]], %[[CREAL]], %[[CIMAG]]) fastmath<contract> : (f32, f32, f32, f32) -> complex<f32>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<complex<f32>>
subroutine div_test_single(a,b,c)
  complex(kind=4) :: a, b, c
  a = b / c
end subroutine div_test_single

! CHECK-LABEL: @_QPdiv_test_double
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<complex<f64>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f64>> {{.*}}, %[[CREF:.*]]: !fir.ref<complex<f64>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<complex<f64>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<complex<f64>>
! CHECK: %[[BREAL:.*]] = fir.extract_value %[[BVAL]], [0 : index] : (complex<f64>) -> f64
! CHECK: %[[BIMAG:.*]] = fir.extract_value %[[BVAL]], [1 : index] : (complex<f64>) -> f64
! CHECK: %[[CREAL:.*]] = fir.extract_value %[[CVAL]], [0 : index] : (complex<f64>) -> f64
! CHECK: %[[CIMAG:.*]] = fir.extract_value %[[CVAL]], [1 : index] : (complex<f64>) -> f64
! CHECK: %[[AVAL:.*]] = fir.call @__divdc3(%[[BREAL]], %[[BIMAG]], %[[CREAL]], %[[CIMAG]]) fastmath<contract> : (f64, f64, f64, f64) -> complex<f64>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<complex<f64>>
subroutine div_test_double(a,b,c)
  complex(kind=8) :: a, b, c
  a = b / c
end subroutine div_test_double

! CHECK-LABEL: @_QPdiv_test_extended
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<complex<f80>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f80>> {{.*}}, %[[CREF:.*]]: !fir.ref<complex<f80>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<complex<f80>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<complex<f80>>
! CHECK: %[[BREAL:.*]] = fir.extract_value %[[BVAL]], [0 : index] : (complex<f80>) -> f80
! CHECK: %[[BIMAG:.*]] = fir.extract_value %[[BVAL]], [1 : index] : (complex<f80>) -> f80
! CHECK: %[[CREAL:.*]] = fir.extract_value %[[CVAL]], [0 : index] : (complex<f80>) -> f80
! CHECK: %[[CIMAG:.*]] = fir.extract_value %[[CVAL]], [1 : index] : (complex<f80>) -> f80
! CHECK: %[[AVAL:.*]] = fir.call @__divxc3(%[[BREAL]], %[[BIMAG]], %[[CREAL]], %[[CIMAG]]) fastmath<contract> : (f80, f80, f80, f80) -> complex<f80>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<complex<f80>>
subroutine div_test_extended(a,b,c)
  complex(kind=10) :: a, b, c
  a = b / c
end subroutine div_test_extended

! CHECK-LABEL: @_QPdiv_test_quad
! CHECK-SAME: %[[AREF:.*]]: !fir.ref<complex<f128>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f128>> {{.*}}, %[[CREF:.*]]: !fir.ref<complex<f128>> {{.*}})
! CHECK: %[[BVAL:.*]] = fir.load %[[BREF]] : !fir.ref<complex<f128>>
! CHECK: %[[CVAL:.*]] = fir.load %[[CREF]] : !fir.ref<complex<f128>>
! CHECK: %[[BREAL:.*]] = fir.extract_value %[[BVAL]], [0 : index] : (complex<f128>) -> f128
! CHECK: %[[BIMAG:.*]] = fir.extract_value %[[BVAL]], [1 : index] : (complex<f128>) -> f128
! CHECK: %[[CREAL:.*]] = fir.extract_value %[[CVAL]], [0 : index] : (complex<f128>) -> f128
! CHECK: %[[CIMAG:.*]] = fir.extract_value %[[CVAL]], [1 : index] : (complex<f128>) -> f128
! CHECK: %[[AVAL:.*]] = fir.call @__divtc3(%[[BREAL]], %[[BIMAG]], %[[CREAL]], %[[CIMAG]]) fastmath<contract> : (f128, f128, f128, f128) -> complex<f128>
! CHECK: fir.store %[[AVAL]] to %[[AREF]] : !fir.ref<complex<f128>>
subroutine div_test_quad(a,b,c)
  complex(kind=16) :: a, b, c
  a = b / c
end subroutine div_test_quad
