! REQUIRES : openmp_runtime

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK: func.func @_QPatomic_implicit_cast_read() {
subroutine atomic_implicit_cast_read
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
end subroutine
