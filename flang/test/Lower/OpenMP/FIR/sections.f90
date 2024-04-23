! REQUIRES: openmp_runtime

! This test checks the lowering of OpenMP sections construct with several clauses present

! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: func @_QQmain() attributes {fir.bindc_name = "sample"} {
!CHECK:   %[[COUNT:.*]] = fir.address_of(@_QFEcount) : !fir.ref<i32>
!CHECK:   %[[ETA:.*]] = fir.alloca f32 {bindc_name = "eta", uniq_name = "_QFEeta"}
!CHECK:   %[[CONST_1:.*]] = arith.constant 4 : i64
!CHECK:   omp.sections allocate(%[[CONST_1]] : i64 -> %0 : !fir.ref<i32>)  {
!CHECK:     omp.section {
!CHECK:       %[[PRIVATE_ETA:.*]] = fir.alloca f32 {bindc_name = "eta", pinned, uniq_name = "_QFEeta"}
!CHECK:       %[[PRIVATE_DOUBLE_COUNT:.*]] = fir.alloca i32 {bindc_name = "double_count", pinned, uniq_name = "_QFEdouble_count"} 
!CHECK:       %[[const:.*]] = arith.constant 5 : i32
!CHECK:       fir.store %[[const]] to %[[COUNT]] : !fir.ref<i32>
!CHECK:       %[[temp_count:.*]] = fir.load %[[COUNT]] : !fir.ref<i32>
!CHECK:       %[[temp_double_count:.*]] = fir.load %[[PRIVATE_DOUBLE_COUNT]] : !fir.ref<i32>
!CHECK:       %[[result:.*]] = arith.muli %[[temp_count]], %[[temp_double_count]] : i32
!CHECK:       {{.*}} = fir.convert %[[result]] : (i32) -> f32
!CHECK:       fir.store {{.*}} to %[[PRIVATE_ETA]] : !fir.ref<f32>
!CHECK:       omp.terminator
!CHECK:     }
!CHECK:     omp.section {
!CHECK:       %[[PRIVATE_ETA:.*]] = fir.alloca f32 {bindc_name = "eta", pinned, uniq_name = "_QFEeta"}
!CHECK:       %[[PRIVATE_DOUBLE_COUNT:.*]] = fir.alloca i32 {bindc_name = "double_count", pinned, uniq_name = "_QFEdouble_count"} 
!CHECK:       %[[temp:.*]] = fir.load %[[PRIVATE_DOUBLE_COUNT]] : !fir.ref<i32>
!CHECK:       %[[const:.*]] = arith.constant 1 : i32
!CHECK:       %[[result:.*]] = arith.addi %[[temp]], %[[const]] : i32
!CHECK:       fir.store %[[result]] to %[[PRIVATE_DOUBLE_COUNT]] : !fir.ref<i32>
!CHECK:       omp.terminator
!CHECK:     }
!CHECK:     omp.section {
!CHECK:       %[[PRIVATE_ETA:.*]] = fir.alloca f32 {bindc_name = "eta", pinned, uniq_name = "_QFEeta"}
!CHECK:       %[[PRIVATE_DOUBLE_COUNT:.*]] = fir.alloca i32 {bindc_name = "double_count", pinned, uniq_name = "_QFEdouble_count"} 
!CHECK:       %[[temp:.*]] = fir.load %[[PRIVATE_ETA]] : !fir.ref<f32>
!CHECK:       %[[const:.*]] = arith.constant 7.000000e+00 : f32
!CHECK:       %[[result:.*]] = arith.subf %[[temp]], %[[const]] {{.*}}: f32
!CHECK:       fir.store %[[result]] to %[[PRIVATE_ETA]] : !fir.ref<f32>
!CHECK:       {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!CHECK:       %[[temp_count:.*]] = fir.convert {{.*}} : (i32) -> f32
!CHECK:       %[[temp_eta:.*]] = fir.load %[[PRIVATE_ETA]] : !fir.ref<f32>
!CHECK:       {{.*}} = arith.mulf %[[temp_count]], %[[temp_eta]] {{.*}}: f32
!CHECK:       %[[result:.*]] = fir.convert {{.*}} : (f32) -> i32
!CHECK:       fir.store %[[result]] to %[[COUNT]] : !fir.ref<i32>
!CHECK:       {{.*}} = fir.load %[[COUNT]] : !fir.ref<i32>
!CHECK:       %[[temp_count:.*]] = fir.convert {{.*}} : (i32) -> f32
!CHECK:       %[[temp_eta:.*]] = fir.load %[[PRIVATE_ETA]] : !fir.ref<f32>
!CHECK:       {{.*}} = arith.subf %[[temp_count]], %[[temp_eta]] {{.*}}: f32
!CHECK:       %[[result:.*]] = fir.convert {{.*}} : (f32) -> i32
!CHECK:       fir.store %[[result]] to %[[PRIVATE_DOUBLE_COUNT]] : !fir.ref<i32>
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
!CHECK:         %[[PRIVATE_ALPHA:.*]] = fir.alloca f32 {bindc_name = "alpha", pinned, uniq_name = "_QFfirstprivateEalpha"}
!CHECK:         %[[temp:.*]] = fir.load %[[ARG]] : !fir.ref<f32>
!CHECK:         fir.store %[[temp]] to %[[PRIVATE_ALPHA]] : !fir.ref<f32>
!CHECK:       omp.terminator
!CHECK:     }
!CHECK:     omp.terminator
!CHECK:   }
!CHECK:   omp.sections {
!CHECK:     omp.section  {
!CHECK:       %[[PRIVATE_VAR:.*]] = fir.load %[[ARG]] : !fir.ref<f32>
!CHECK:       %[[CONSTANT:.*]] = arith.constant 5.000000e+00 : f32
!CHECK:       %[[PRIVATE_VAR_2:.*]] = arith.mulf %[[PRIVATE_VAR]], %[[CONSTANT]] {{.*}}: f32
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

subroutine lastprivate()
	integer :: x
!CHECK: %[[X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFlastprivateEx"}
!CHECK: omp.sections   {
	!$omp sections lastprivate(x)
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivateEx"}
!CHECK: %[[const:.*]] = arith.constant 10 : i32
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.muli %c10_i32, %[[temp]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
        !$omp section
            x = x * 10
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivateEx"}
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 1 : i32
!CHECK: %[[result:.*]] = arith.addi %[[temp]], %[[const]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[true:.*]] = arith.constant true
!CHECK: fir.if %[[true]] {
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[X]] : !fir.ref<i32>
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
        !$omp section
            x = x + 1
!CHECK: omp.terminator
!CHECK: }
    !$omp end sections
    
!CHECK: omp.sections   {
    !$omp sections firstprivate(x) lastprivate(x)
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivateEx"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[const:.*]] = arith.constant 10 : i32
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.muli %c10_i32, %[[temp]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
        !$omp section
            x = x * 10
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivateEx"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 1 : i32
!CHECK: %[[result:.*]] = arith.addi %[[temp]], %[[const]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[true:.*]] = arith.constant true
!CHECK: fir.if %true {
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[X]] : !fir.ref<i32>
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
        !$omp section
            x = x + 1
!CHECK: omp.terminator
!CHECK: }
    !$omp end sections

!CHECK: omp.sections nowait {
    !$omp sections firstprivate(x) lastprivate(x)
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivateEx"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[const:.*]] = arith.constant 10 : i32
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[result:.*]] = arith.muli %c10_i32, %[[temp]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
        !$omp section
            x = x * 10
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivateEx"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 1 : i32
!CHECK: %[[result:.*]] = arith.addi %[[temp]], %[[const]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[true:.*]] = arith.constant true
!CHECK: fir.if %true {
!CHECK: %[[temp:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: fir.store %[[temp]] to %[[X]] : !fir.ref<i32>
!CHECK: omp.barrier
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
        !$omp section
            x = x + 1
!CHECK: omp.terminator
!CHECK: }
    !$omp end sections nowait

!CHECK: omp.sections {
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivateEx"}
!CHECK: cf.br ^bb1
!CHECK: ^bb1:  // pred: ^bb0
!CHECK: %[[INNER_PRIVATE_X:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[const:.*]] = arith.constant 1 : i32
!CHECK: %[[result:.*]] = arith.addi %[[INNER_PRIVATE_X]], %[[const]] : i32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: %[[loaded_value:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<i32>
!CHECK: fir.store %[[loaded_value]] to %[[X]] : !fir.ref<i32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
!CHECK: return
!CHECK: }

    !$omp sections lastprivate(x)
        !$omp section
                goto 30
        30  x = x + 1
    !$omp end sections
end subroutine

subroutine unstructured_sections_privatization()
!CHECK: %[[X:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFunstructured_sections_privatizationEx"}
!CHECK: omp.sections {
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFunstructured_sections_privatizationEx"}
!CHECK: cf.br ^bb1
!CHECK: ^bb1:  // pred: ^bb0
!CHECK: %[[INNER_PRIVATE_X:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<f32>
!CHECK: %[[constant:.*]] = arith.constant 1.000000e+00 : f32
!CHECK: %[[result:.*]] = arith.addf %[[INNER_PRIVATE_X]], %[[constant]] fastmath<contract> : f32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<f32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
    !$omp sections private(x)
        !$omp section
            goto 40
        40  x = x + 1
    !$omp end sections
!CHECK: omp.sections {
!CHECK: omp.section {
!CHECK: %[[PRIVATE_X:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFunstructured_sections_privatizationEx"}
!CHECK: %[[temp:.*]] = fir.load %[[X]] : !fir.ref<f32>
!CHECK: fir.store %[[temp]] to %[[PRIVATE_X]] : !fir.ref<f32>
!CHECK: cf.br ^bb1
!CHECK: ^bb1:  // pred: ^bb0
!CHECK: %[[INNER_PRIVATE_X:.*]] = fir.load %[[PRIVATE_X]] : !fir.ref<f32>
!CHECK: %[[constant:.*]] = arith.constant 1.000000e+00 : f32
!CHECK: %[[result:.*]] = arith.addf %[[INNER_PRIVATE_X]], %[[constant]] fastmath<contract> : f32
!CHECK: fir.store %[[result]] to %[[PRIVATE_X]] : !fir.ref<f32>
!CHECK: omp.terminator
!CHECK: }
!CHECK: omp.terminator
!CHECK: }
    !$omp sections firstprivate(x)
        !$omp section
            goto 50
        50  x = x + 1
    !$omp end sections
end subroutine
