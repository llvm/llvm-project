! This test checks the lowering of OpenMP sections construct with several clauses present

! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

!CHECK: func @_QQmain() {
!CHECK:   %[[COUNT:.*]] = fir.address_of(@_QFEcount) : !fir.ref<i32>
!CHECK:   %[[DOUBLE_COUNT:.*]] = fir.address_of(@_QFEdouble_count) : !fir.ref<i32>
!CHECK:   %[[ETA:.*]] = fir.alloca f32 {bindc_name = "eta", uniq_name = "_QFEeta"}
!CHECK:   %[[CONST_1:.*]] = arith.constant 1 : i32
!CHECK:   omp.sections allocate(%[[CONST_1]] : i32 -> %0 : !fir.ref<i32>)  {
!CHECK:     omp.section {
!CHECK:       {{.*}} = arith.constant 5 : i32
!CHECK:       fir.store {{.*}} to {{.*}} : !fir.ref<i32>
!CHECK:       {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!CHECK:       {{.*}} = fir.load %[[DOUBLE_COUNT]] : !fir.ref<i32>
!CHECK:       {{.*}} = arith.muli {{.*}}, {{.*}} : i32
!CHECK:       {{.*}} = fir.convert {{.*}} : (i32) -> f32
!CHECK:       fir.store {{.*}} to %[[ETA]] : !fir.ref<f32>
!CHECK:       omp.terminator
!CHECK:     }
!CHECK:     omp.section {
!CHECK:       {{.*}} = fir.load %[[DOUBLE_COUNT]] : !fir.ref<i32>
!CHECK:       {{.*}} = arith.constant 1 : i32
!CHECK:       {{.*}} = arith.addi {{.*}} : i32
!CHECK:       fir.store {{.*}} to %[[DOUBLE_COUNT]] : !fir.ref<i32>
!CHECK:       omp.terminator
!CHECK:     }
!CHECK:     omp.section {
!CHECK:       {{.*}} = fir.load %[[ETA]] : !fir.ref<f32>
!CHECK:       {{.*}} = arith.constant 7.000000e+00 : f32
!CHECK:       {{.*}} = arith.subf {{.*}} : f32
!CHECK:       fir.store {{.*}} to %[[ETA]] : !fir.ref<f32>
!CHECK:       {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!CHECK:       {{.*}} = fir.convert {{.*}} : (i32) -> f32
!CHECK:       {{.*}} = fir.load %[[ETA]] : !fir.ref<f32>
!CHECK:       {{.*}} = arith.mulf {{.*}}, {{.*}} : f32
!CHECK:       {{.*}} = fir.convert {{.*}} : (f32) -> i32
!CHECK:       fir.store {{.*}} to %[[COUNT]] : !fir.ref<i32>
!CHECK:       {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!CHECK:       {{.*}} = fir.convert {{.*}} : (i32) -> f32
!CHECK:       {{.*}} = fir.load %[[ETA]] : !fir.ref<f32>
!CHECK:       {{.*}} = arith.subf {{.*}}, {{.*}} : f32
!CHECK:       {{.*}} = fir.convert {{.*}} : (f32) -> i32
!CHECK:       fir.store {{.*}} to %[[DOUBLE_COUNT]] : !fir.ref<i32>
!CHECK:       omp.terminator
!CHECK:     }
!CHECK:     omp.terminator
!CHECK:   }
!CHECK:   omp.sections nowait {
!CHECK:     omp.terminator
!CHECK:   }
!CHECK:   return
!CHECK: }

program sample
    use omp_lib
    integer :: count = 0, double_count = 1
    !$omp sections private (eta, double_count) allocate(omp_high_bw_mem_alloc: count)
        !$omp section
            count = 1 + 4
            eta = count * double_count
        !$omp section
            double_count = double_count + 1
        !$omp section
            eta = eta - 7
            count = count * eta
            double_count = count - eta
    !$omp end sections

    !$omp sections
    !$omp end sections nowait
end program sample

!CHECK: func @_QPfirstprivate(%[[ARG:.*]]: !fir.ref<f32> {fir.bindc_name = "alpha"}) {
!CHECK:   omp.sections {
!CHECK:     omp.section  {
!CHECK:       omp.terminator
!CHECK:     }
!CHECK:     omp.terminator
!CHECK:   }
!CHECK:   omp.sections {
!CHECK:     omp.section  {
!CHECK:       %[[PRIVATE_VAR:.*]] = fir.load %[[ARG]] : !fir.ref<f32>
!CHECK:       %[[CONSTANT:.*]] = arith.constant 5.000000e+00 : f32
!CHECK:       %[[PRIVATE_VAR_2:.*]] = arith.mulf %[[PRIVATE_VAR]], %[[CONSTANT]] : f32
!CHECK:       fir.store %[[PRIVATE_VAR_2]] to %[[ARG]] : !fir.ref<f32>
!CHECK:       omp.terminator
!CHECK:     }
!CHECK:     omp.terminator
!CHECK:   }
!CHECK:   return
!CHECK: }

subroutine firstprivate(alpha)
    real :: alpha
    !$omp sections firstprivate(alpha)
    !$omp end sections

    !$omp sections
        alpha = alpha * 5
    !$omp end sections
end subroutine
