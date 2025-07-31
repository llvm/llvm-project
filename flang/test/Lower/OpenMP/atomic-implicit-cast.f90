! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK: func.func @_QPatomic_implicit_cast_read() {
subroutine atomic_implicit_cast_read
! CHECK: %[[ALLOCA7:.*]] = fir.alloca complex<f64>
! CHECK: %[[ALLOCA6:.*]] = fir.alloca i32
! CHECK: %[[ALLOCA5:.*]] = fir.alloca i32
! CHECK: %[[ALLOCA4:.*]] = fir.alloca i32
! CHECK: %[[ALLOCA3:.*]] = fir.alloca complex<f32>
! CHECK: %[[ALLOCA2:.*]] = fir.alloca complex<f32>
! CHECK: %[[ALLOCA1:.*]] = fir.alloca i32
! CHECK: %[[ALLOCA0:.*]] = fir.alloca f32

! CHECK: %[[M:.*]] = fir.alloca complex<f64> {bindc_name = "m", uniq_name = "_QFatomic_implicit_cast_readEm"}
! CHECK: %[[M_DECL:.*]]:2 = hlfir.declare %[[M]] {uniq_name = "_QFatomic_implicit_cast_readEm"} : (!fir.ref<complex<f64>>) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
! CHECK: %[[W:.*]] = fir.alloca complex<f32> {bindc_name = "w", uniq_name = "_QFatomic_implicit_cast_readEw"}
! CHECK: %[[W_DECL:.*]]:2 = hlfir.declare %[[W]] {uniq_name = "_QFatomic_implicit_cast_readEw"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFatomic_implicit_cast_readEx"}
! CHECK: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFatomic_implicit_cast_readEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[Y:.*]] = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFatomic_implicit_cast_readEy"}
! CHECK: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]] {uniq_name = "_QFatomic_implicit_cast_readEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[Z:.*]] = fir.alloca f64 {bindc_name = "z", uniq_name = "_QFatomic_implicit_cast_readEz"}
! CHECK: %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]] {uniq_name = "_QFatomic_implicit_cast_readEz"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    integer :: x
    real :: y
    double precision :: z
    complex :: w
    complex(8) :: m

! CHECK: omp.atomic.read %[[ALLOCA0:.*]] = %[[Y_DECL]]#0 : !fir.ref<f32>, !fir.ref<f32>, f32
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA0]] : !fir.ref<f32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (f32) -> i32
! CHECK: fir.store %[[CVT]] to %[[X_DECL]]#0 : !fir.ref<i32>
    !$omp atomic read
        x = y

! CHECK: omp.atomic.read %[[ALLOCA1:.*]] = %[[X_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA1]] : !fir.ref<i32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (i32) -> f64
! CHECK: fir.store %[[CVT]] to %[[Z_DECL]]#0 : !fir.ref<f64>
    !$omp atomic read
        z = x

! CHECK: omp.atomic.read %[[ALLOCA2:.*]] = %[[W_DECL]]#0 : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA2]] : !fir.ref<complex<f32>>
! CHECK: %[[EXTRACT:.*]] = fir.extract_value %[[LOAD]], [0 : index] : (complex<f32>) -> f32
! CHECK: %[[CVT:.*]] = fir.convert %[[EXTRACT]] : (f32) -> i32
! CHECK: fir.store %[[CVT]] to %[[X_DECL]]#0 : !fir.ref<i32>
    !$omp atomic read
        x = w

! CHECK: omp.atomic.read %[[ALLOCA3:.*]] = %[[W_DECL]]#0 : !fir.ref<complex<f32>>, !fir.ref<complex<f32>>, complex<f32>
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA3]] : !fir.ref<complex<f32>>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (complex<f32>) -> complex<f64>
! CHECK: fir.store %[[CVT]] to %[[M_DECL]]#0 : !fir.ref<complex<f64>>
    !$omp atomic read
        m = w

! CHECK: %[[CONST:.*]] = arith.constant 1 : i32
! CHECK: omp.atomic.capture {
! CHECK: omp.atomic.read %[[ALLOCA4]] = %[[X_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK: omp.atomic.update %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK: ^bb0(%[[ARG:.*]]: i32):
! CHECK: %[[RESULT:.*]] = arith.addi %[[ARG]], %[[CONST]] : i32
! CHECK: omp.yield(%[[RESULT]] : i32)
! CHECK: }
! CHECK: }
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA4]] : !fir.ref<i32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (i32) -> f32
! CHECK: fir.store %[[CVT]] to %[[Y_DECL]]#0 : !fir.ref<f32>
     !$omp atomic capture
        y = x
        x = x + 1
     !$omp end atomic

! CHECK: %[[CONST:.*]] = arith.constant 10 : i32
! CHECK: omp.atomic.capture {
! CHECK: omp.atomic.read %[[ALLOCA5:.*]] = %[[X_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK: omp.atomic.write %[[X_DECL]]#0 = %[[CONST]] : !fir.ref<i32>, i32
! CHECK: }
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA5]] : !fir.ref<i32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (i32) -> f64
! CHECK: fir.store %[[CVT]] to %[[Z_DECL]]#0 : !fir.ref<f64>
     !$omp atomic capture
        z = x
        x = 10
     !$omp end atomic

! CHECK: %[[CONST:.*]] = arith.constant 1 : i32
! CHECK: omp.atomic.capture {
! CHECK: omp.atomic.update %[[X_DECL]]#0 : !fir.ref<i32> {
! CHECK: ^bb0(%[[ARG:.*]]: i32):
! CHECK: %[[RESULT:.*]] = arith.addi %[[ARG]], %[[CONST]] : i32
! CHECK: omp.yield(%[[RESULT]] : i32)
! CHECK: }
! CHECK: omp.atomic.read %[[ALLOCA6]] = %[[X_DECL]]#0 : !fir.ref<i32>, !fir.ref<i32>, i32
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA6]] : !fir.ref<i32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (i32) -> f32
! CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
! CHECK: %[[UNDEF:.*]] = fir.undefined complex<f32>
! CHECK: %[[IDX1:.*]] = fir.insert_value %[[UNDEF]], %[[CVT]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK: %[[IDX2:.*]] = fir.insert_value %[[IDX1]], %[[CST]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK: fir.store %[[IDX2]] to %[[W_DECL]]#0 : !fir.ref<complex<f32>>
     !$omp atomic capture
        x = x + 1
        w = x
     !$omp end atomic


! CHECK: %[[CST1:.*]] = arith.constant 1.000000e+00 : f64
! CHECK: %[[CST2:.*]] = arith.constant 0.000000e+00 : f64
! CHECK: %[[UNDEF:.*]] = fir.undefined complex<f64>
! CHECK: %[[IDX1:.*]] = fir.insert_value %[[UNDEF]], %[[CST1]], [0 : index] : (complex<f64>, f64) -> complex<f64>
! CHECK: %[[IDX2:.*]] = fir.insert_value %[[IDX1]], %[[CST2]], [1 : index] : (complex<f64>, f64) -> complex<f64>
! CHECK: omp.atomic.capture {
! CHECK: omp.atomic.update %[[M_DECL]]#0 : !fir.ref<complex<f64>> {
! CHECK: ^bb0(%[[ARG:.*]]: complex<f64>):
! CHECK: %[[RESULT:.*]] = fir.addc %[[ARG]], %[[IDX2]] {fastmath = #arith.fastmath<contract>} : complex<f64>
! CHECK: omp.yield(%[[RESULT]] : complex<f64>)
! CHECK: }
! CHECK: omp.atomic.read %[[ALLOCA7]] = %[[M_DECL]]#0 : !fir.ref<complex<f64>>, !fir.ref<complex<f64>>, complex<f64>
! CHECK: }
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA7]] : !fir.ref<complex<f64>>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (complex<f64>) -> complex<f32>
! CHECK: fir.store %[[CVT]] to %[[W_DECL]]#0 : !fir.ref<complex<f32>>
     !$omp atomic capture
        m = m + 1
        w = m
     !$omp end atomic


end subroutine
