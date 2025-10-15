!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes=HOST,ALL
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=ALL

PROGRAM main
    ! HOST-DAG: %[[I_REF:.*]] = fir.alloca f32 {bindc_name = "i", uniq_name = "_QFEi"}
    ! HOST-DAG: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_REF]] {uniq_name = "_QFEi"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
    REAL :: I
    ! ALL-DAG: fir.global internal @_QFEi {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>} : f32 {
    ! ALL-DAG: %[[UNDEF:.*]] = fir.zero_bits f32
    ! ALL-DAG: fir.has_value %[[UNDEF]] : f32
    ! ALL-DAG: }
    !$omp declare target(I)
END
