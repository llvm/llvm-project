! REQUIRES: openmp_runtime

! RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - | FileCheck %s

! CHECK: func.func @_QPatomic_implicit_cast_read() {
subroutine atomic_implicit_cast_read
! CHECK: %[[VAL_M:.*]] = fir.alloca complex<f64> {bindc_name = "m", uniq_name = "_QFatomic_implicit_cast_readEm"}
! CHECK: %[[VAL_M_DECLARE:.*]]:2 = hlfir.declare %[[VAL_M]] {uniq_name = "_QFatomic_implicit_cast_readEm"} : (!fir.ref<complex<f64>>) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
! CHECK: %[[VAL_W:.*]] = fir.alloca complex<f32> {bindc_name = "w", uniq_name = "_QFatomic_implicit_cast_readEw"}
! CHECK: %[[VAL_W_DECLARE:.*]]:2 = hlfir.declare %[[VAL_W]] {uniq_name = "_QFatomic_implicit_cast_readEw"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK: %[[VAL_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFatomic_implicit_cast_readEx"}
! CHECK: %[[VAL_X_DECLARE:.*]]:2 = hlfir.declare %[[VAL_X]] {uniq_name = "_QFatomic_implicit_cast_readEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[VAL_Y:.*]] = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFatomic_implicit_cast_readEy"}
! CHECK: %[[VAL_Y_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Y]] {uniq_name = "_QFatomic_implicit_cast_readEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[VAL_Z:.*]] = fir.alloca f64 {bindc_name = "z", uniq_name = "_QFatomic_implicit_cast_readEz"}
! CHECK: %[[VAL_Z_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Z]] {uniq_name = "_QFatomic_implicit_cast_readEz"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    integer :: x
    real    :: y
    double precision :: z
    complex :: w
    complex(8) :: m

    ! Atomic read

! CHECK: %[[ALLOCA:.*]] = fir.alloca i32
! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_Y_DECLARE]]#1 : !fir.ref<f32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (f32) -> i32
! CHECK: fir.store %[[CVT]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK: omp.atomic.read %[[VAL_X_DECLARE]]#1 = %[[ALLOCA]] : !fir.ref<i32>, !fir.ref<i32>, i32
    !$omp atomic read
       x = y

! CHECK: %[[ALLOCA:.*]] = fir.alloca f64
! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_X_DECLARE]]#1 : !fir.ref<i32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (i32) -> f64
! CHECK: fir.store %[[CVT]] to %[[ALLOCA]] : !fir.ref<f64>
! CHECK: omp.atomic.read %[[VAL_Z_DECLARE]]#1 = %[[ALLOCA]] : !fir.ref<f64>, !fir.ref<f64>, f64
    !$omp atomic read
       z = x

! CHECK: %[[ALLOCA:.*]] = fir.alloca i32
! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_W_DECLARE]]#1 : !fir.ref<complex<f32>>
! CHECK: %[[EXT:.*]] = fir.extract_value %[[LOAD]], [0 : index] : (complex<f32>) -> f32
! CHECK: %[[CVT:.*]] = fir.convert %[[EXT]] : (f32) -> i32
! CHECK: fir.store %[[CVT]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK: omp.atomic.read %[[VAL_X_DECLARE]]#1 = %[[ALLOCA]] : !fir.ref<i32>, !fir.ref<i32>, i32
    !$omp atomic read
       x = w

! CHECK: %[[ALLOCA:.*]] = fir.alloca f32
! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_W_DECLARE]]#1 : !fir.ref<complex<f32>>
! CHECK: %[[EXT:.*]] = fir.extract_value %[[LOAD]], [0 : index] : (complex<f32>) -> f32
! CHECK: %[[CVT:.*]] = fir.convert %[[EXT]] : (f32) -> f32
! CHECK: fir.store %[[CVT]] to %[[ALLOCA]] : !fir.ref<f32>
! CHECK: omp.atomic.read %[[VAL_Y_DECLARE]]#1 = %[[ALLOCA]] : !fir.ref<f32>, !fir.ref<f32>, f32
    !$omp atomic read
       y = w

! CHECK: %[[ALLOCA:.*]] = fir.alloca complex<f64>
! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_W_DECLARE]]#1 : !fir.ref<complex<f32>>
! CHECK: %[[EXT0:.*]] = fir.extract_value %[[LOAD]], [0 : index] : (complex<f32>) -> f32
! CHECK: %[[EXT1:.*]] = fir.extract_value %[[LOAD]], [1 : index] : (complex<f32>) -> f32
! CHECK: %[[CVT0:.*]] = fir.convert %[[EXT0]] : (f32) -> f64
! CHECK: %[[CVT1:.*]] = fir.convert %[[EXT1]] : (f32) -> f64
! CHECK: %[[UNDEF:.*]] = fir.undefined complex<f64>
! CHECK: %[[INSERT1:.*]] = fir.insert_value %[[UNDEF]], %[[CVT0]], [0 : index] : (complex<f64>, f64) -> complex<f64>
! CHECK: %[[INSERT2:.*]] = fir.insert_value %[[INSERT1]], %[[CVT1]], [1 : index] : (complex<f64>, f64) -> complex<f64>
! CHECK: fir.store %[[INSERT2]] to %[[ALLOCA]] : !fir.ref<complex<f64>>
! CHECK: omp.atomic.read %[[VAL_M_DECLARE]]#1 = %[[ALLOCA]] : !fir.ref<complex<f64>>, !fir.ref<complex<f64>>, complex<f64>
    !$omp atomic read
       m = w
end subroutine
! CHECK: func.func @_QPatomic_implicit_cast_write()
subroutine atomic_implicit_cast_write
! CHECK: %[[VAL_M:.*]] = fir.alloca complex<f64> {bindc_name = "m", uniq_name = "_QFatomic_implicit_cast_writeEm"}
! CHECK: %[[VAL_M_DECLARE:.*]]:2 = hlfir.declare %[[VAL_M]] {uniq_name = "_QFatomic_implicit_cast_writeEm"} : (!fir.ref<complex<f64>>) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
! CHECK: %[[VAL_W:.*]] = fir.alloca complex<f32> {bindc_name = "w", uniq_name = "_QFatomic_implicit_cast_writeEw"}
! CHECK: %[[VAL_W_DECLARE:.*]]:2 = hlfir.declare %[[VAL_W]] {uniq_name = "_QFatomic_implicit_cast_writeEw"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK: %[[VAL_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFatomic_implicit_cast_writeEx"}
! CHECK: %[[VAL_X_DECLARE:.*]]:2 = hlfir.declare %[[VAL_X]] {uniq_name = "_QFatomic_implicit_cast_writeEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[VAL_Y:.*]] = fir.alloca f32 {bindc_name = "y", uniq_name = "_QFatomic_implicit_cast_writeEy"}
! CHECK: %[[VAL_Y_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Y]] {uniq_name = "_QFatomic_implicit_cast_writeEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK: %[[VAL_Z:.*]] = fir.alloca f64 {bindc_name = "z", uniq_name = "_QFatomic_implicit_cast_writeEz"}
! CHECK: %[[VAL_Z_DECLARE:.*]]:2 = hlfir.declare %[[VAL_Z]] {uniq_name = "_QFatomic_implicit_cast_writeEz"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
    integer :: x
    real    :: y
    double precision :: z
    complex :: w
    complex(8) :: m
 
! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_Y_DECLARE]]#0 : !fir.ref<f32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (f32) -> i32
! CHECK: omp.atomic.write %[[VAL_X_DECLARE]]#1 = %[[CVT]] : !fir.ref<i32>, i32
    !$omp atomic write
       x = y

! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_X_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (i32) -> f64
! CHECK: omp.atomic.write %[[VAL_Z_DECLARE:.*]] = %[[CVT]] : !fir.ref<f64>, f64
    !$omp atomic write
       z = x

! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_W_DECLARE]]#0 : !fir.ref<complex<f32>>
! CHECK: %[[EXT:.*]] = fir.extract_value %[[LOAD]], [0 : index] : (complex<f32>) -> f32
! CHECK: %[[CVT:.*]] = fir.convert %[[EXT]] : (f32) -> i32
! CHECK: omp.atomic.write %[[VAL_X_DECLARE]]#1 = %[[CVT]] : !fir.ref<i32>, i32
    !$omp atomic write
       x = w

! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_W_DECLARE]]#0 : !fir.ref<complex<f32>>
! CHECK: %[[EXT:.*]] = fir.extract_value %[[LOAD]], [0 : index] : (complex<f32>) -> f32
! CHECK: omp.atomic.write %[[VAL_Y_DECLARE]]#1 = %[[EXT]] : !fir.ref<f32>, f32
    !$omp atomic write
       y = w 
 
! CHECK: %[[LOAD:.*]] = fir.load %[[VAL_W_DECLARE]]#0 : !fir.ref<complex<f32>>
! CHECK: %[[CVT:.*]] = fir.convert %[[LOAD]] : (complex<f32>) -> complex<f64>
! CHECK: omp.atomic.write %[[VAL_M_DECLARE]]#1 = %[[CVT]] : !fir.ref<complex<f64>>, complex<f64>
    !$omp atomic write
       m = w
end subroutine
