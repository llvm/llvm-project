!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s 
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

PROGRAM main
    ! CHECK-DAG: %0 = fir.alloca f32 {bindc_name = "i", uniq_name = "_QFEi"}
    REAL :: I
    ! CHECK-DAG: fir.global internal @_QFEi {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32 {
    ! CHECK-DAG: %0 = fir.undefined f32
    ! CHECK-DAG: fir.has_value %0 : f32
    ! CHECK-DAG: }
    !$omp declare target(I)
END
