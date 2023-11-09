! This test checks lowering of `FIRSTPRIVATE` clause for scalar types.

! REQUIRES: shell
! RUN: bbc -fopenmp -emit-fir -hlfir=false %s -o - | FileCheck %s --check-prefix=FIRDialect

!FIRDialect-DAG: func @_QPfirstprivate_complex(%[[ARG1:.*]]: !fir.ref<!fir.complex<4>>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.complex<8>>{{.*}}) {
!FIRDialect:   omp.parallel {
!FIRDialect:     %[[ARG1_PVT:.*]] = fir.alloca !fir.complex<4> {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_complexEarg1"}
!FIRDialect:     %[[ARG1_VAL:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.complex<4>>
!FIRDialect:     fir.store %[[ARG1_VAL]] to %[[ARG1_PVT]] : !fir.ref<!fir.complex<4>>
!FIRDialect:     %[[ARG2_PVT:.*]] = fir.alloca !fir.complex<8> {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_complexEarg2"}
!FIRDialect:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2]] : !fir.ref<!fir.complex<8>>
!FIRDialect:     fir.store %[[ARG2_VAL]] to %[[ARG2_PVT]] : !fir.ref<!fir.complex<8>>
!FIRDialect:     fir.call @_QPfoo(%[[ARG1_PVT]], %[[ARG2_PVT]]) {{.*}}: (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<8>>) -> ()
!FIRDialect:     omp.terminator
!FIRDialect:   }

subroutine firstprivate_complex(arg1, arg2)
        complex(4) :: arg1
        complex(8) :: arg2

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2)
        call foo(arg1, arg2)
!$OMP END PARALLEL

end subroutine

!FIRDialect-DAG: func @_QPfirstprivate_integer(%[[ARG1:.*]]: !fir.ref<i32>{{.*}}, %[[ARG2:.*]]: !fir.ref<i8>{{.*}}, %[[ARG3:.*]]: !fir.ref<i16>{{.*}}, %[[ARG4:.*]]: !fir.ref<i32>{{.*}}, %[[ARG5:.*]]: !fir.ref<i64>{{.*}}, %[[ARG6:.*]]: !fir.ref<i128>{{.*}}) {
!FIRDialect:  omp.parallel {
!FIRDialect:    %[[ARG1_PVT:.*]] = fir.alloca i32 {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_integerEarg1"}
!FIRDialect:    %[[ARG1_VAL:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
!FIRDialect:    fir.store %[[ARG1_VAL]] to %[[ARG1_PVT]] : !fir.ref<i32>
!FIRDialect:    %[[ARG2_PVT:.*]] = fir.alloca i8 {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_integerEarg2"}
!FIRDialect:    %[[ARG2_VAL:.*]] = fir.load %[[ARG2]] : !fir.ref<i8>
!FIRDialect:    fir.store %[[ARG2_VAL]] to %[[ARG2_PVT]] : !fir.ref<i8>
!FIRDialect:    %[[ARG3_PVT:.*]] = fir.alloca i16 {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_integerEarg3"}
!FIRDialect:    %[[ARG3_VAL:.*]] = fir.load %[[ARG3]] : !fir.ref<i16>
!FIRDialect:    fir.store %[[ARG3_VAL]] to %[[ARG3_PVT]] : !fir.ref<i16>
!FIRDialect:    %[[ARG4_PVT:.*]] = fir.alloca i32 {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_integerEarg4"}
!FIRDialect:    %[[ARG4_VAL:.*]] = fir.load %[[ARG4]] : !fir.ref<i32>
!FIRDialect:    fir.store %[[ARG4_VAL]] to %[[ARG4_PVT]] : !fir.ref<i32>
!FIRDialect:    %[[ARG5_PVT:.*]] = fir.alloca i64 {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_integerEarg5"}
!FIRDialect:    %[[ARG5_VAL:.*]] = fir.load %[[ARG5]] : !fir.ref<i64>
!FIRDialect:    fir.store %[[ARG5_VAL]] to %[[ARG5_PVT]] : !fir.ref<i64>
!FIRDialect:    %[[ARG6_PVT:.*]] = fir.alloca i128 {bindc_name = "arg6", pinned, uniq_name = "_QFfirstprivate_integerEarg6"}
!FIRDialect:    %[[ARG6_VAL:.*]] = fir.load %[[ARG6]] : !fir.ref<i128>
!FIRDialect:    fir.store %[[ARG6_VAL]] to %[[ARG6_PVT]] : !fir.ref<i128>
!FIRDialect:    fir.call @_QPbar(%[[ARG1_PVT]], %[[ARG2_PVT]], %[[ARG3_PVT]], %[[ARG4_PVT]], %[[ARG5_PVT]], %[[ARG6_PVT]]) {{.*}}: (!fir.ref<i32>, !fir.ref<i8>, !fir.ref<i16>, !fir.ref<i32>, !fir.ref<i64>, !fir.ref<i128>) -> ()
!FIRDialect:    omp.terminator
!FIRDialect:  }

subroutine firstprivate_integer(arg1, arg2, arg3, arg4, arg5, arg6)
        integer :: arg1
        integer(kind=1) :: arg2
        integer(kind=2) :: arg3
        integer(kind=4) :: arg4
        integer(kind=8) :: arg5
        integer(kind=16) :: arg6

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2, arg3, arg4, arg5, arg6)
        call bar(arg1, arg2, arg3, arg4, arg5, arg6)
!$OMP END PARALLEL

end subroutine

!FIRDialect-DAG: func @_QPfirstprivate_logical(%[[ARG1:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.logical<1>>{{.*}}, %[[ARG3:.*]]: !fir.ref<!fir.logical<2>>{{.*}}, %[[ARG4:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[ARG5:.*]]: !fir.ref<!fir.logical<8>>{{.*}}) {
!FIRDialect:   omp.parallel {
!FIRDialect:     %[[ARG1_PVT:.*]] = fir.alloca !fir.logical<4> {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_logicalEarg1"}
!FIRDialect:     %[[ARG1_VAL:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.logical<4>>
!FIRDialect:     fir.store %[[ARG1_VAL]] to %[[ARG1_PVT]] : !fir.ref<!fir.logical<4>>
!FIRDialect:     %[[ARG2_PVT:.*]] = fir.alloca !fir.logical<1> {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_logicalEarg2"}
!FIRDialect:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2]] : !fir.ref<!fir.logical<1>>
!FIRDialect:     fir.store %[[ARG2_VAL]] to %[[ARG2_PVT]] : !fir.ref<!fir.logical<1>>
!FIRDialect:     %[[ARG3_PVT:.*]] = fir.alloca !fir.logical<2> {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_logicalEarg3"}
!FIRDialect:     %[[ARG3_VAL:.*]] = fir.load %[[ARG3]] : !fir.ref<!fir.logical<2>>
!FIRDialect:     fir.store %[[ARG3_VAL]] to %[[ARG3_PVT]] : !fir.ref<!fir.logical<2>>
!FIRDialect:     %[[ARG4_PVT:.*]] = fir.alloca !fir.logical<4> {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_logicalEarg4"}
!FIRDialect:     %[[ARG4_VAL:.*]] = fir.load %[[ARG4]] : !fir.ref<!fir.logical<4>>
!FIRDialect:     fir.store %[[ARG4_VAL]] to %[[ARG4_PVT]] : !fir.ref<!fir.logical<4>>
!FIRDialect:     %[[ARG5_PVT:.*]] = fir.alloca !fir.logical<8> {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_logicalEarg5"}
!FIRDialect:     %[[ARG5_VAL:.*]] = fir.load %[[ARG5]] : !fir.ref<!fir.logical<8>>
!FIRDialect:     fir.store %[[ARG5_VAL]] to %[[ARG5_PVT]] : !fir.ref<!fir.logical<8>>
!FIRDialect:     fir.call @_QPbaz(%[[ARG1_PVT]], %[[ARG2_PVT]], %[[ARG3_PVT]], %[[ARG4_PVT]], %[[ARG5_PVT]]) {{.*}}: (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<1>>, !fir.ref<!fir.logical<2>>, !fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<8>>) -> ()
!FIRDialect:     omp.terminator
!FIRDialect:   }

subroutine firstprivate_logical(arg1, arg2, arg3, arg4, arg5)
        logical :: arg1
        logical(kind=1) :: arg2
        logical(kind=2) :: arg3
        logical(kind=4) :: arg4
        logical(kind=8) :: arg5

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2, arg3, arg4, arg5)
        call baz(arg1, arg2, arg3, arg4, arg5)
!$OMP END PARALLEL

end subroutine

!FIRDialect-DAG: func @_QPfirstprivate_real(%[[ARG1:.*]]: !fir.ref<f32>{{.*}}, %[[ARG2:.*]]: !fir.ref<f16>{{.*}}, %[[ARG3:.*]]: !fir.ref<f32>{{.*}}, %[[ARG4:.*]]: !fir.ref<f64>{{.*}}, %[[ARG5:.*]]: !fir.ref<f80>{{.*}}, %[[ARG6:.*]]: !fir.ref<f128>{{.*}}) {
!FIRDialect:   omp.parallel {
!FIRDialect:     %[[ARG1_PVT:.*]] = fir.alloca f32 {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_realEarg1"}
!FIRDialect:     %[[ARG1_VAL:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
!FIRDialect:     fir.store %[[ARG1_VAL]] to %[[ARG1_PVT]] : !fir.ref<f32>
!FIRDialect:     %[[ARG2_PVT:.*]] = fir.alloca f16 {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_realEarg2"}
!FIRDialect:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2]] : !fir.ref<f16>
!FIRDialect:     fir.store %[[ARG2_VAL]] to %[[ARG2_PVT]] : !fir.ref<f16>
!FIRDialect:     %[[ARG3_PVT:.*]] = fir.alloca f32 {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_realEarg3"}
!FIRDialect:     %[[ARG3_VAL:.*]] = fir.load %[[ARG3]] : !fir.ref<f32>
!FIRDialect:     fir.store %[[ARG3_VAL]] to %[[ARG3_PVT]] : !fir.ref<f32>
!FIRDialect:     %[[ARG4_PVT:.*]] = fir.alloca f64 {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_realEarg4"}
!FIRDialect:     %[[ARG4_VAL:.*]] = fir.load %[[ARG4]] : !fir.ref<f64>
!FIRDialect:     fir.store %[[ARG4_VAL]] to %[[ARG4_PVT]] : !fir.ref<f64>
!FIRDialect:     %[[ARG5_PVT:.*]] = fir.alloca f80 {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_realEarg5"}
!FIRDialect:     %[[ARG5_VAL:.*]] = fir.load %[[ARG5]] : !fir.ref<f80>
!FIRDialect:     fir.store %[[ARG5_VAL]] to %[[ARG5_PVT]] : !fir.ref<f80>
!FIRDialect:     %[[ARG6_PVT:.*]] = fir.alloca f128 {bindc_name = "arg6", pinned, uniq_name = "_QFfirstprivate_realEarg6"}
!FIRDialect:     %[[ARG6_VAL:.*]] = fir.load %[[ARG6]] : !fir.ref<f128>
!FIRDialect:     fir.store %[[ARG6_VAL]] to %[[ARG6_PVT]] : !fir.ref<f128>
!FIRDialect:     fir.call @_QPqux(%[[ARG1_PVT]], %[[ARG2_PVT]], %[[ARG3_PVT]], %[[ARG4_PVT]], %[[ARG5_PVT]], %[[ARG6_PVT]]) {{.*}}: (!fir.ref<f32>, !fir.ref<f16>, !fir.ref<f32>, !fir.ref<f64>, !fir.ref<f80>, !fir.ref<f128>) -> ()
!FIRDialect:     omp.terminator
!FIRDialect:   }

subroutine firstprivate_real(arg1, arg2, arg3, arg4, arg5, arg6)
        real :: arg1
        real(kind=2) :: arg2
        real(kind=4) :: arg3
        real(kind=8) :: arg4
        real(kind=10) :: arg5
        real(kind=16) :: arg6

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2, arg3, arg4, arg5, arg6)
        call qux(arg1, arg2, arg3, arg4, arg5, arg6)
!$OMP END PARALLEL

end subroutine

!FIRDialect-LABEL:   func.func @_QPmultiple_firstprivate(
!FIRDialect-SAME:                                        %[[A_ADDR:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
!FIRDialect-SAME:                                        %[[B_ADDR:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}) {
!FIRDialect:           omp.parallel   {
!FIRDialect:             %[[A_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFmultiple_firstprivateEa"}
!FIRDialect:             %[[A:.*]] = fir.load %[[A_ADDR]] : !fir.ref<i32>
!FIRDialect:             fir.store %[[A]] to %[[A_PRIV_ADDR]] : !fir.ref<i32>
!FIRDialect:             %[[B_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "b", pinned, uniq_name = "_QFmultiple_firstprivateEb"}
!FIRDialect:             %[[B:.*]] = fir.load %[[B_ADDR]] : !fir.ref<i32>
!FIRDialect:             fir.store %[[B]] to %[[B_PRIV_ADDR]] : !fir.ref<i32>
!FIRDialect:             fir.call @_QPquux(%[[A_PRIV_ADDR]], %[[B_PRIV_ADDR]]) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
!FIRDialect:             omp.terminator
!FIRDialect:           }
!FIRDialect:           return
!FIRDialect:         }

subroutine multiple_firstprivate(a, b)
        integer :: a, b
!$OMP PARALLEL FIRSTPRIVATE(a) FIRSTPRIVATE(b)
        call quux(a, b)
!$OMP END PARALLEL
end subroutine multiple_firstprivate
