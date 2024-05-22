!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes=HOST,ALL
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=ALL

PROGRAM main
    ! HOST-DAG: %0 = fir.alloca f32 {bindc_name = "i", uniq_name = "_QFEi"}
    REAL :: I
    ! ALL-DAG: fir.global internal @_QFEi {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32 {
    ! ALL-DAG: %0 = fir.undefined f32
    ! ALL-DAG: fir.has_value %0 : f32
    ! ALL-DAG: }
    !$omp declare target(I)
END
