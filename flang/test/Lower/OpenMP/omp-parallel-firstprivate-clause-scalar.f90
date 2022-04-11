! This test checks lowering of `FIRSTPRIVATE` clause for scalar types.

! REQUIRES: shell
! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QPfirstprivate_complex(%[[ARG1:.*]]: !fir.ref<!fir.complex<4>>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.complex<8>>{{.*}}) {
!FIRDialect-DAG:   omp.parallel {
!FIRDialect-DAG:     %[[ARG1_PVT:.*]] = fir.alloca !fir.complex<4> {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_complexEarg1"}
!FIRDialect-DAG:     %[[ARG1_VAL:.*]] = fir.load %arg0 : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:     fir.store %[[ARG1_VAL]] to %[[ARG1_PVT]] : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:     %[[ARG2_PVT:.*]] = fir.alloca !fir.complex<8> {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_complexEarg2"}
!FIRDialect-DAG:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2]] : !fir.ref<!fir.complex<8>>
!FIRDialect-DAG:     fir.store %[[ARG2_VAL]] to %[[ARG2_PVT]] : !fir.ref<!fir.complex<8>>
!FIRDialect-DAG:     %[[LIST_IO:.*]] = fir.call @_FortranAioBeginExternalListOutput
!FIRDialect-DAG:     %[[ARG1_PVT_VAL:.*]] = fir.load %[[ARG1_PVT]] : !fir.ref<!fir.complex<4>>
!FIRDialect-DAG:     %[[ARG1_PVT_REAL:.*]] = fir.extract_value %[[ARG1_PVT_VAL]], [0 : index] : (!fir.complex<4>) -> f32
!FIRDialect-DAG:     %[[ARG1_PVT_IMAG:.*]] = fir.extract_value %[[ARG1_PVT_VAL]], [1 : index] : (!fir.complex<4>) -> f32
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputComplex32(%[[LIST_IO]], %[[ARG1_PVT_REAL]], %[[ARG1_PVT_IMAG]]) : (!fir.ref<i8>, f32, f32) -> i1
!FIRDialect-DAG:     %[[ARG2_PVT_VAL:.*]] = fir.load %[[ARG2_PVT]] : !fir.ref<!fir.complex<8>>
!FIRDialect-DAG:     %[[ARG2_PVT_REAL:.*]] = fir.extract_value %[[ARG2_PVT_VAL]], [0 : index] : (!fir.complex<8>) -> f64
!FIRDialect-DAG:     %[[ARG2_PVT_IMAG:.*]] = fir.extract_value %[[ARG2_PVT_VAL]], [1 : index] : (!fir.complex<8>) -> f64
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputComplex64(%[[LIST_IO]], %[[ARG2_PVT_REAL]], %[[ARG2_PVT_IMAG]]) : (!fir.ref<i8>, f64, f64) -> i1
!FIRDialect-DAG:     omp.terminator
!FIRDialect-DAG:   }

subroutine firstprivate_complex(arg1, arg2)
        complex(4) :: arg1
        complex(8) :: arg2

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2)
        print *, arg1, arg2
!$OMP END PARALLEL

end subroutine

!FIRDialect: func @_QPfirstprivate_integer(%[[ARG1:.*]]: !fir.ref<i32>{{.*}}, %[[ARG2:.*]]: !fir.ref<i8>{{.*}}, %[[ARG3:.*]]: !fir.ref<i16>{{.*}}, %[[ARG4:.*]]: !fir.ref<i32>{{.*}}, %[[ARG5:.*]]: !fir.ref<i64>{{.*}}, %[[ARG6:.*]]: !fir.ref<i128>{{.*}}) {
!FIRDialect-DAG:  omp.parallel {
!FIRDialect-DAG:    %[[ARG1_PVT:.*]] = fir.alloca i32 {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_integerEarg1"}
!FIRDialect-DAG:    %[[ARG1_VAL:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
!FIRDialect-DAG:    fir.store %[[ARG1_VAL]] to %[[ARG1_PVT]] : !fir.ref<i32>
!FIRDialect-DAG:    %[[ARG2_PVT:.*]] = fir.alloca i8 {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_integerEarg2"}
!FIRDialect-DAG:    %[[ARG2_VAL:.*]] = fir.load %[[ARG2]] : !fir.ref<i8>
!FIRDialect-DAG:    fir.store %[[ARG2_VAL]] to %[[ARG2_PVT]] : !fir.ref<i8>
!FIRDialect-DAG:    %[[ARG3_PVT:.*]] = fir.alloca i16 {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_integerEarg3"}
!FIRDialect-DAG:    %[[ARG3_VAL:.*]] = fir.load %[[ARG3]] : !fir.ref<i16>
!FIRDialect-DAG:    fir.store %[[ARG3_VAL]] to %[[ARG3_PVT]] : !fir.ref<i16>
!FIRDialect-DAG:    %[[ARG4_PVT:.*]] = fir.alloca i32 {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_integerEarg4"}
!FIRDialect-DAG:    %[[ARG4_VAL:.*]] = fir.load %[[ARG4]] : !fir.ref<i32>
!FIRDialect-DAG:    fir.store %[[ARG4_VAL]] to %[[ARG4_PVT]] : !fir.ref<i32>
!FIRDialect-DAG:    %[[ARG5_PVT:.*]] = fir.alloca i64 {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_integerEarg5"}
!FIRDialect-DAG:    %[[ARG5_VAL:.*]] = fir.load %[[ARG5]] : !fir.ref<i64>
!FIRDialect-DAG:    fir.store %[[ARG5_VAL]] to %[[ARG5_PVT]] : !fir.ref<i64>
!FIRDialect-DAG:    %[[ARG6_PVT:.*]] = fir.alloca i128 {bindc_name = "arg6", pinned, uniq_name = "_QFfirstprivate_integerEarg6"}
!FIRDialect-DAG:    %[[ARG6_VAL:.*]] = fir.load %[[ARG6]] : !fir.ref<i128>
!FIRDialect-DAG:    fir.store %[[ARG6_VAL]] to %[[ARG6_PVT]] : !fir.ref<i128>
!FIRDialect-DAG:    %[[LIST_IO:.*]] = fir.call @_FortranAioBeginExternalListOutput
!FIRDialect-DAG:    %[[ARG1_PVT_VAL:.*]] = fir.load %[[ARG1_PVT]] : !fir.ref<i32>
!FIRDialect-DAG:    %{{.*}} = fir.call @_FortranAioOutputInteger32(%[[LIST_IO]], %[[ARG1_PVT_VAL]]) : (!fir.ref<i8>, i32) -> i1
!FIRDialect-DAG:    %[[ARG2_PVT_VAL:.*]] = fir.load %[[ARG2_PVT]] : !fir.ref<i8>
!FIRDialect-DAG:    %{{.*}} = fir.call @_FortranAioOutputInteger8(%[[LIST_IO]], %[[ARG2_PVT_VAL]]) : (!fir.ref<i8>, i8) -> i1
!FIRDialect-DAG:    %[[ARG3_PVT_VAL:.*]] = fir.load %[[ARG3_PVT]] : !fir.ref<i16>
!FIRDialect-DAG:    %{{.*}} = fir.call @_FortranAioOutputInteger16(%[[LIST_IO]], %[[ARG3_PVT_VAL]]) : (!fir.ref<i8>, i16) -> i1
!FIRDialect-DAG:    %[[ARG4_PVT_VAL:.*]] = fir.load %[[ARG4_PVT]] : !fir.ref<i32>
!FIRDialect-DAG:    %{{.*}} = fir.call @_FortranAioOutputInteger32(%[[LIST_IO]], %[[ARG4_PVT_VAL]]) : (!fir.ref<i8>, i32) -> i1
!FIRDialect-DAG:    %[[ARG5_PVT_VAL:.*]] = fir.load %[[ARG5_PVT]] : !fir.ref<i64>
!FIRDialect-DAG:    %{{.*}} = fir.call @_FortranAioOutputInteger64(%[[LIST_IO]], %[[ARG5_PVT_VAL]]) : (!fir.ref<i8>, i64) -> i1
!FIRDialect-DAG:    %[[ARG6_PVT_VAL:.*]] = fir.load %[[ARG6_PVT]] : !fir.ref<i128>
!FIRDialect-DAG:    %{{.*}} = fir.call @_FortranAioOutputInteger128(%[[LIST_IO]], %[[ARG6_PVT_VAL]]) : (!fir.ref<i8>, i128) -> i1
!FIRDialect-DAG:    omp.terminator
!FIRDialect-DAG:  }

subroutine firstprivate_integer(arg1, arg2, arg3, arg4, arg5, arg6)
        integer :: arg1
        integer(kind=1) :: arg2
        integer(kind=2) :: arg3
        integer(kind=4) :: arg4
        integer(kind=8) :: arg5
        integer(kind=16) :: arg6

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2, arg3, arg4, arg5, arg6)
        print *, arg1, arg2, arg3, arg4, arg5, arg6
!$OMP END PARALLEL

end subroutine

!FIRDialect: func @_QPfirstprivate_logical(%[[ARG1:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.logical<1>>{{.*}}, %[[ARG3:.*]]: !fir.ref<!fir.logical<2>>{{.*}}, %[[ARG4:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[ARG5:.*]]: !fir.ref<!fir.logical<8>>{{.*}}) {
!FIRDialect-DAG:   omp.parallel {
!FIRDialect-DAG:     %[[ARG1_PVT:.*]] = fir.alloca !fir.logical<4> {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_logicalEarg1"}
!FIRDialect-DAG:     %[[ARG1_VAL:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:     fir.store %[[ARG1_VAL]] to %[[ARG1_PVT]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:     %[[ARG2_PVT:.*]] = fir.alloca !fir.logical<1> {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_logicalEarg2"}
!FIRDialect-DAG:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2]] : !fir.ref<!fir.logical<1>>
!FIRDialect-DAG:     fir.store %[[ARG2_VAL]] to %[[ARG2_PVT]] : !fir.ref<!fir.logical<1>>
!FIRDialect-DAG:     %[[ARG3_PVT:.*]] = fir.alloca !fir.logical<2> {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_logicalEarg3"}
!FIRDialect-DAG:     %[[ARG3_VAL:.*]] = fir.load %[[ARG3]] : !fir.ref<!fir.logical<2>>
!FIRDialect-DAG:     fir.store %[[ARG3_VAL]] to %[[ARG3_PVT]] : !fir.ref<!fir.logical<2>>
!FIRDialect-DAG:     %[[ARG4_PVT:.*]] = fir.alloca !fir.logical<4> {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_logicalEarg4"}
!FIRDialect-DAG:     %[[ARG4_VAL:.*]] = fir.load %[[ARG4]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:     fir.store %[[ARG4_VAL]] to %[[ARG4_PVT]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:     %[[ARG5_PVT:.*]] = fir.alloca !fir.logical<8> {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_logicalEarg5"}
!FIRDialect-DAG:     %[[ARG5_VAL:.*]] = fir.load %[[ARG5]] : !fir.ref<!fir.logical<8>>
!FIRDialect-DAG:     fir.store %[[ARG5_VAL]] to %[[ARG5_PVT]] : !fir.ref<!fir.logical<8>>
!FIRDialect-DAG:     %[[LIST_IO:.*]] = fir.call @_FortranAioBeginExternalListOutput
!FIRDialect-DAG:     %[[ARG1_PVT_VAL:.*]] = fir.load %[[ARG1_PVT]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:     %[[ARG1_PVT_CVT:.*]] = fir.convert %[[ARG1_PVT_VAL]] : (!fir.logical<4>) -> i1
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputLogical(%[[LIST_IO]], %[[ARG1_PVT_CVT]]) : (!fir.ref<i8>, i1) -> i1
!FIRDialect-DAG:     %[[ARG2_PVT_VAL:.*]] = fir.load %[[ARG2_PVT]] : !fir.ref<!fir.logical<1>>
!FIRDialect-DAG:     %[[ARG2_PVT_CVT:.*]] = fir.convert %[[ARG2_PVT_VAL]] : (!fir.logical<1>) -> i1
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputLogical(%[[LIST_IO]], %[[ARG2_PVT_CVT]]) : (!fir.ref<i8>, i1) -> i1
!FIRDialect-DAG:     %[[ARG3_PVT_VAL:.*]] = fir.load %[[ARG3_PVT]] : !fir.ref<!fir.logical<2>>
!FIRDialect-DAG:     %[[ARG3_PVT_CVT:.*]] = fir.convert %[[ARG3_PVT_VAL]] : (!fir.logical<2>) -> i1
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputLogical(%[[LIST_IO]], %[[ARG3_PVT_CVT]]) : (!fir.ref<i8>, i1) -> i1
!FIRDialect-DAG:     %[[ARG4_PVT_VAL:.*]] = fir.load %[[ARG4_PVT]] : !fir.ref<!fir.logical<4>>
!FIRDialect-DAG:     %[[ARG4_PVT_CVT:.*]] = fir.convert %[[ARG4_PVT_VAL]] : (!fir.logical<4>) -> i1
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputLogical(%[[LIST_IO]], %[[ARG4_PVT_CVT]]) : (!fir.ref<i8>, i1) -> i1
!FIRDialect-DAG:     %[[ARG5_PVT_VAL:.*]] = fir.load %[[ARG5_PVT]] : !fir.ref<!fir.logical<8>>
!FIRDialect-DAG:     %[[ARG5_PVT_CVT:.*]] = fir.convert %[[ARG5_PVT_VAL]] : (!fir.logical<8>) -> i1
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputLogical(%[[LIST_IO]], %[[ARG5_PVT_CVT]]) : (!fir.ref<i8>, i1) -> i1
!FIRDialect-DAG:     omp.terminator
!FIRDialect-DAG:   }

subroutine firstprivate_logical(arg1, arg2, arg3, arg4, arg5)
        logical :: arg1
        logical(kind=1) :: arg2
        logical(kind=2) :: arg3
        logical(kind=4) :: arg4
        logical(kind=8) :: arg5

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2, arg3, arg4, arg5)
        print *, arg1, arg2, arg3, arg4, arg5
!$OMP END PARALLEL

end subroutine

!FIRDialect-DAG: func @_QPfirstprivate_real(%[[ARG1:.*]]: !fir.ref<f32>{{.*}}, %[[ARG2:.*]]: !fir.ref<f16>{{.*}}, %[[ARG3:.*]]: !fir.ref<f32>{{.*}}, %[[ARG4:.*]]: !fir.ref<f64>{{.*}}, %[[ARG5:.*]]: !fir.ref<f80>{{.*}}, %[[ARG6:.*]]: !fir.ref<f128>{{.*}}) {
!FIRDialect-DAG:   omp.parallel {
!FIRDialect-DAG:     %[[ARG1_PVT:.*]] = fir.alloca f32 {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_realEarg1"}
!FIRDialect-DAG:     %[[ARG1_VAL:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
!FIRDialect-DAG:     fir.store %[[ARG1_VAL]] to %[[ARG1_PVT]] : !fir.ref<f32>
!FIRDialect-DAG:     %[[ARG2_PVT:.*]] = fir.alloca f16 {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_realEarg2"}
!FIRDialect-DAG:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2]] : !fir.ref<f16>
!FIRDialect-DAG:     fir.store %[[ARG2_VAL]] to %[[ARG2_PVT]] : !fir.ref<f16>
!FIRDialect-DAG:     %[[ARG3_PVT:.*]] = fir.alloca f32 {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_realEarg3"}
!FIRDialect-DAG:     %[[ARG3_VAL:.*]] = fir.load %[[ARG3]] : !fir.ref<f32>
!FIRDialect-DAG:     fir.store %[[ARG3_VAL]] to %[[ARG3_PVT]] : !fir.ref<f32>
!FIRDialect-DAG:     %[[ARG4_PVT:.*]] = fir.alloca f64 {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_realEarg4"}
!FIRDialect-DAG:     %[[ARG4_VAL:.*]] = fir.load %[[ARG4]] : !fir.ref<f64>
!FIRDialect-DAG:     fir.store %[[ARG4_VAL]] to %[[ARG4_PVT]] : !fir.ref<f64>
!FIRDialect-DAG:     %[[ARG5_PVT:.*]] = fir.alloca f80 {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_realEarg5"}
!FIRDialect-DAG:     %[[ARG5_VAL:.*]] = fir.load %[[ARG5]] : !fir.ref<f80>
!FIRDialect-DAG:     fir.store %[[ARG5_VAL]] to %[[ARG5_PVT]] : !fir.ref<f80>
!FIRDialect-DAG:     %[[ARG6_PVT:.*]] = fir.alloca f128 {bindc_name = "arg6", pinned, uniq_name = "_QFfirstprivate_realEarg6"}
!FIRDialect-DAG:     %[[ARG6_VAL:.*]] = fir.load %[[ARG6]] : !fir.ref<f128>
!FIRDialect-DAG:     fir.store %[[ARG6_VAL]] to %[[ARG6_PVT]] : !fir.ref<f128>
!FIRDialect-DAG:     %[[LIST_IO:.*]] = fir.call @_FortranAioBeginExternalListOutput
!FIRDialect-DAG:     %[[ARG1_PVT_VAL:.*]] = fir.load %[[ARG1_PVT]] : !fir.ref<f32>
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputReal32(%[[LIST_IO]], %[[ARG1_PVT_VAL]]) : (!fir.ref<i8>, f32) -> i1
!FIRDialect-DAG:     %[[ARG2_PVT_VAL:.*]] = fir.embox %[[ARG2_PVT]] : (!fir.ref<f16>) -> !fir.box<f16>
!FIRDialect-DAG:     %[[ARG2_PVT_CVT:.*]] = fir.convert %[[ARG2_PVT_VAL]] : (!fir.box<f16>) -> !fir.box<none>
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputDescriptor(%[[LIST_IO]], %[[ARG2_PVT_CVT]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
!FIRDialect-DAG:     %[[ARG3_PVT_VAL:.*]] = fir.load %[[ARG3_PVT]] : !fir.ref<f32>
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputReal32(%[[LIST_IO]], %[[ARG3_PVT_VAL]]) : (!fir.ref<i8>, f32) -> i1
!FIRDialect-DAG:     %[[ARG4_PVT_VAL:.*]] = fir.load %[[ARG4_PVT]] : !fir.ref<f64>
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputReal64(%[[LIST_IO]], %[[ARG4_PVT_VAL]]) : (!fir.ref<i8>, f64) -> i1
!FIRDialect-DAG:     %[[ARG5_PVT_VAL:.*]] = fir.embox %[[ARG5_PVT]] : (!fir.ref<f80>) -> !fir.box<f80>
!FIRDialect-DAG:     %[[ARG5_PVT_CVT:.*]] = fir.convert %[[ARG5_PVT_VAL]] : (!fir.box<f80>) -> !fir.box<none>
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputDescriptor(%[[LIST_IO]], %[[ARG5_PVT_CVT]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
!FIRDialect-DAG:     %[[ARG6_PVT_VAL:.*]] = fir.embox %[[ARG6_PVT]] : (!fir.ref<f128>) -> !fir.box<f128>
!FIRDialect-DAG:     %[[ARG6_PVT_CVT:.*]] = fir.convert %[[ARG6_PVT_VAL]] : (!fir.box<f128>) -> !fir.box<none>
!FIRDialect-DAG:     %{{.*}} = fir.call @_FortranAioOutputDescriptor(%[[LIST_IO]], %[[ARG6_PVT_CVT]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
!FIRDialect-DAG:     omp.terminator
!FIRDialect-DAG:   }

subroutine firstprivate_real(arg1, arg2, arg3, arg4, arg5, arg6)
        real :: arg1
        real(kind=2) :: arg2
        real(kind=4) :: arg3
        real(kind=8) :: arg4
        real(kind=10) :: arg5
        real(kind=16) :: arg6

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2, arg3, arg4, arg5, arg6)
        print *, arg1, arg2, arg3, arg4, arg5, arg6
!$OMP END PARALLEL

end subroutine
