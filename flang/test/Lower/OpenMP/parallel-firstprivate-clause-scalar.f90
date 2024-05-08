! This test checks lowering of `FIRSTPRIVATE` clause for scalar types.

! REQUIRES: shell
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s --check-prefix=CHECK

!CHECK-DAG: func @_QPfirstprivate_complex(%[[ARG1:.*]]: !fir.ref<!fir.complex<4>>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.complex<8>>{{.*}}) {
!CHECK:    %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_complexEarg1"} : (!fir.ref<!fir.complex<4>>, !fir.dscope) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
!CHECK:    %[[ARG2_DECL:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_complexEarg2"} : (!fir.ref<!fir.complex<8>>, !fir.dscope) -> (!fir.ref<!fir.complex<8>>, !fir.ref<!fir.complex<8>>)
!CHECK:   omp.parallel {
!CHECK:     %[[ARG1_PVT:.*]] = fir.alloca !fir.complex<4> {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_complexEarg1"}
!CHECK:     %[[ARG1_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG1_PVT]] {uniq_name = "_QFfirstprivate_complexEarg1"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
!CHECK:     %[[ARG1_VAL:.*]] = fir.load %[[ARG1_DECL]]#0 : !fir.ref<!fir.complex<4>>
!CHECK:      hlfir.assign %[[ARG1_VAL]] to %[[ARG1_PVT_DECL]]#0 temporary_lhs : !fir.complex<4>, !fir.ref<!fir.complex<4>>
!CHECK:     %[[ARG2_PVT:.*]] = fir.alloca !fir.complex<8> {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_complexEarg2"}
!CHECK:     %[[ARG2_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG2_PVT]] {uniq_name = "_QFfirstprivate_complexEarg2"} : (!fir.ref<!fir.complex<8>>) -> (!fir.ref<!fir.complex<8>>, !fir.ref<!fir.complex<8>>)
!CHECK:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2_DECL]]#0 : !fir.ref<!fir.complex<8>>
!CHECK:     hlfir.assign %[[ARG2_VAL]] to %[[ARG2_PVT_DECL]]#0 temporary_lhs : !fir.complex<8>, !fir.ref<!fir.complex<8>>
!CHECK:     fir.call @_QPfoo(%[[ARG1_PVT_DECL]]#1, %[[ARG2_PVT_DECL]]#1) {{.*}}: (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<8>>) -> ()
!CHECK:     omp.terminator
!CHECK:   }

subroutine firstprivate_complex(arg1, arg2)
        complex(4) :: arg1
        complex(8) :: arg2

!$OMP PARALLEL FIRSTPRIVATE(arg1, arg2)
        call foo(arg1, arg2)
!$OMP END PARALLEL

end subroutine

!CHECK-DAG: func @_QPfirstprivate_integer(%[[ARG1:.*]]: !fir.ref<i32>{{.*}}, %[[ARG2:.*]]: !fir.ref<i8>{{.*}}, %[[ARG3:.*]]: !fir.ref<i16>{{.*}}, %[[ARG4:.*]]: !fir.ref<i32>{{.*}}, %[[ARG5:.*]]: !fir.ref<i64>{{.*}}, %[[ARG6:.*]]: !fir.ref<i128>{{.*}}) {
!CHECK:  %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_integerEarg1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[ARG2_DECL:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_integerEarg2"} : (!fir.ref<i8>, !fir.dscope) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK:  %[[ARG3_DECL:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_integerEarg3"} : (!fir.ref<i16>, !fir.dscope) -> (!fir.ref<i16>, !fir.ref<i16>)
!CHECK:  %[[ARG4_DECL:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_integerEarg4"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[ARG5_DECL:.*]]:2 = hlfir.declare %[[ARG5]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_integerEarg5"} : (!fir.ref<i64>, !fir.dscope) -> (!fir.ref<i64>, !fir.ref<i64>)
!CHECK:  %[[ARG6_DECL:.*]]:2 = hlfir.declare %[[ARG6]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_integerEarg6"} : (!fir.ref<i128>, !fir.dscope) -> (!fir.ref<i128>, !fir.ref<i128>)
!CHECK:  omp.parallel {
!CHECK:    %[[ARG1_PVT:.*]] = fir.alloca i32 {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_integerEarg1"}
!CHECK:    %[[ARG1_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG1_PVT]] {uniq_name = "_QFfirstprivate_integerEarg1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[ARG1_VAL:.*]] = fir.load %[[ARG1_DECL]]#0 : !fir.ref<i32>
!CHECK:    hlfir.assign %[[ARG1_VAL]] to %[[ARG1_PVT_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:    %[[ARG2_PVT:.*]] = fir.alloca i8 {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_integerEarg2"}
!CHECK:    %[[ARG2_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG2_PVT]] {uniq_name = "_QFfirstprivate_integerEarg2"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK:    %[[ARG2_VAL:.*]] = fir.load %[[ARG2_DECL]]#0 : !fir.ref<i8>
!CHECK:    hlfir.assign %[[ARG2_VAL]] to %[[ARG2_PVT_DECL]]#0 temporary_lhs : i8, !fir.ref<i8>
!CHECK:    %[[ARG3_PVT:.*]] = fir.alloca i16 {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_integerEarg3"}
!CHECK:    %[[ARG3_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG3_PVT]] {uniq_name = "_QFfirstprivate_integerEarg3"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!CHECK:    %[[ARG3_VAL:.*]] = fir.load %[[ARG3_DECL]]#0 : !fir.ref<i16>
!CHECK:    hlfir.assign %[[ARG3_VAL]] to %[[ARG3_PVT_DECL]]#0 temporary_lhs : i16, !fir.ref<i16>
!CHECK:    %[[ARG4_PVT:.*]] = fir.alloca i32 {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_integerEarg4"}
!CHECK:    %[[ARG4_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG4_PVT]] {uniq_name = "_QFfirstprivate_integerEarg4"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[ARG4_VAL:.*]] = fir.load %[[ARG4_DECL]]#0 : !fir.ref<i32>
!CHECK:    hlfir.assign %[[ARG4_VAL]] to %[[ARG4_PVT_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:    %[[ARG5_PVT:.*]] = fir.alloca i64 {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_integerEarg5"}
!CHECK:    %[[ARG5_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG5_PVT]] {uniq_name = "_QFfirstprivate_integerEarg5"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!CHECK:    %[[ARG5_VAL:.*]] = fir.load %[[ARG5_DECL]]#0 : !fir.ref<i64>
!CHECK:    hlfir.assign %[[ARG5_VAL]] to %[[ARG5_PVT_DECL]]#0 temporary_lhs : i64, !fir.ref<i64>
!CHECK:    %[[ARG6_PVT:.*]] = fir.alloca i128 {bindc_name = "arg6", pinned, uniq_name = "_QFfirstprivate_integerEarg6"}
!CHECK:    %[[ARG6_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG6_PVT]] {uniq_name = "_QFfirstprivate_integerEarg6"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
!CHECK:    %[[ARG6_VAL:.*]] = fir.load %[[ARG6_DECL]]#0 : !fir.ref<i128>
!CHECK:    hlfir.assign %[[ARG6_VAL]] to %[[ARG6_PVT_DECL]]#0 temporary_lhs : i128, !fir.ref<i128>
!CHECK:    fir.call @_QPbar(%[[ARG1_PVT_DECL]]#1, %[[ARG2_PVT_DECL]]#1, %[[ARG3_PVT_DECL]]#1, %[[ARG4_PVT_DECL]]#1,
!%[[ARG5_PVT_DECL]]#1, %[[ARG6_PVT_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i8>, !fir.ref<i16>, !fir.ref<i32>, !fir.ref<i64>, !fir.ref<i128>) -> ()
!CHECK:    omp.terminator
!CHECK:  }

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

!CHECK-DAG: func @_QPfirstprivate_logical(%[[ARG1:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.logical<1>>{{.*}}, %[[ARG3:.*]]: !fir.ref<!fir.logical<2>>{{.*}}, %[[ARG4:.*]]: !fir.ref<!fir.logical<4>>{{.*}}, %[[ARG5:.*]]: !fir.ref<!fir.logical<8>>{{.*}}) {
!CHECK:    %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_logicalEarg1"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:    %[[ARG2_DECL:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_logicalEarg2"} : (!fir.ref<!fir.logical<1>>, !fir.dscope) -> (!fir.ref<!fir.logical<1>>, !fir.ref<!fir.logical<1>>)
!CHECK:    %[[ARG3_DECL:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_logicalEarg3"} : (!fir.ref<!fir.logical<2>>, !fir.dscope) -> (!fir.ref<!fir.logical<2>>, !fir.ref<!fir.logical<2>>)
!CHECK:    %[[ARG4_DECL:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_logicalEarg4"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:    %[[ARG5_DECL:.*]]:2 = hlfir.declare %[[ARG5]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_logicalEarg5"} : (!fir.ref<!fir.logical<8>>, !fir.dscope) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>)
!CHECK:   omp.parallel {
!CHECK:     %[[ARG1_PVT:.*]] = fir.alloca !fir.logical<4> {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_logicalEarg1"}
!CHECK:     %[[ARG1_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG1_PVT]] {uniq_name = "_QFfirstprivate_logicalEarg1"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:     %[[ARG1_VAL:.*]] = fir.load %[[ARG1_DECL]]#0 : !fir.ref<!fir.logical<4>>
!CHECK:     hlfir.assign %[[ARG1_VAL]] to %[[ARG1_PVT_DECL]]#0 temporary_lhs : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:     %[[ARG2_PVT:.*]] = fir.alloca !fir.logical<1> {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_logicalEarg2"}
!CHECK:     %[[ARG2_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG2_PVT]] {uniq_name = "_QFfirstprivate_logicalEarg2"} : (!fir.ref<!fir.logical<1>>) -> (!fir.ref<!fir.logical<1>>, !fir.ref<!fir.logical<1>>)
!CHECK:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2_DECL]]#0 : !fir.ref<!fir.logical<1>>
!CHECK:     hlfir.assign %[[ARG2_VAL]] to %[[ARG2_PVT_DECL]]#0 temporary_lhs : !fir.logical<1>, !fir.ref<!fir.logical<1>>
!CHECK:     %[[ARG3_PVT:.*]] = fir.alloca !fir.logical<2> {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_logicalEarg3"}
!CHECK:     %[[ARG3_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG3_PVT]] {uniq_name = "_QFfirstprivate_logicalEarg3"} : (!fir.ref<!fir.logical<2>>) -> (!fir.ref<!fir.logical<2>>, !fir.ref<!fir.logical<2>>)
!CHECK:     %[[ARG3_VAL:.*]] = fir.load %[[ARG3_DECL]]#0 : !fir.ref<!fir.logical<2>>
!CHECK:     hlfir.assign %[[ARG3_VAL]] to %[[ARG3_PVT_DECL]]#0 temporary_lhs : !fir.logical<2>, !fir.ref<!fir.logical<2>>
!CHECK:     %[[ARG4_PVT:.*]] = fir.alloca !fir.logical<4> {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_logicalEarg4"}
!CHECK:     %[[ARG4_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG4_PVT]] {uniq_name = "_QFfirstprivate_logicalEarg4"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!CHECK:     %[[ARG4_VAL:.*]] = fir.load %[[ARG4_DECL]]#0 : !fir.ref<!fir.logical<4>>
!CHECK:     hlfir.assign %[[ARG4_VAL]] to %[[ARG4_PVT_DECL]]#0 temporary_lhs : !fir.logical<4>, !fir.ref<!fir.logical<4>>
!CHECK:     %[[ARG5_PVT:.*]] = fir.alloca !fir.logical<8> {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_logicalEarg5"}
!CHECK:     %[[ARG5_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG5_PVT]] {uniq_name = "_QFfirstprivate_logicalEarg5"} : (!fir.ref<!fir.logical<8>>) -> (!fir.ref<!fir.logical<8>>, !fir.ref<!fir.logical<8>>)
!CHECK:     %[[ARG5_VAL:.*]] = fir.load %[[ARG5_DECL]]#0 : !fir.ref<!fir.logical<8>>
!CHECK:     hlfir.assign %[[ARG5_VAL]] to %[[ARG5_PVT_DECL]]#0 temporary_lhs : !fir.logical<8>, !fir.ref<!fir.logical<8>>
!CHECK:     fir.call @_QPbaz(%[[ARG1_PVT_DECL]]#1, %[[ARG2_PVT_DECL]]#1, %[[ARG3_PVT_DECL]]#1, %[[ARG4_PVT_DECL]]#1, %[[ARG5_PVT_DECL]]#1) {{.*}}: (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<1>>, !fir.ref<!fir.logical<2>>, !fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<8>>) -> ()
!CHECK:     omp.terminator
!CHECK:   }

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

!CHECK-DAG: func @_QPfirstprivate_real(%[[ARG1:.*]]: !fir.ref<f32>{{.*}}, %[[ARG2:.*]]: !fir.ref<f16>{{.*}}, %[[ARG3:.*]]: !fir.ref<f32>{{.*}}, %[[ARG4:.*]]: !fir.ref<f64>{{.*}}, %[[ARG5:.*]]: !fir.ref<f80>{{.*}}, %[[ARG6:.*]]: !fir.ref<f128>{{.*}}) {
!CHECK:   %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_realEarg1"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:   %[[ARG2_DECL:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_realEarg2"} : (!fir.ref<f16>, !fir.dscope) -> (!fir.ref<f16>, !fir.ref<f16>)
!CHECK:   %[[ARG3_DECL:.*]]:2 = hlfir.declare %[[ARG3]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_realEarg3"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:   %[[ARG4_DECL:.*]]:2 = hlfir.declare %[[ARG4]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_realEarg4"} : (!fir.ref<f64>, !fir.dscope) -> (!fir.ref<f64>, !fir.ref<f64>)
!CHECK:   %[[ARG5_DECL:.*]]:2 = hlfir.declare %[[ARG5]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_realEarg5"} : (!fir.ref<f80>, !fir.dscope) -> (!fir.ref<f80>, !fir.ref<f80>)
!CHECK:   %[[ARG6_DECL:.*]]:2 = hlfir.declare %[[ARG6]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFfirstprivate_realEarg6"} : (!fir.ref<f128>, !fir.dscope) -> (!fir.ref<f128>, !fir.ref<f128>)
!CHECK:   omp.parallel {
!CHECK:     %[[ARG1_PVT:.*]] = fir.alloca f32 {bindc_name = "arg1", pinned, uniq_name = "_QFfirstprivate_realEarg1"}
!CHECK:     %[[ARG1_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG1_PVT]] {uniq_name = "_QFfirstprivate_realEarg1"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:     %[[ARG1_VAL:.*]] = fir.load %[[ARG1_DECL]]#0 : !fir.ref<f32>
!CHECK:     hlfir.assign %[[ARG1_VAL]] to %[[ARG1_PVT_DECL]]#0 temporary_lhs : f32, !fir.ref<f32>
!CHECK:     %[[ARG2_PVT:.*]] = fir.alloca f16 {bindc_name = "arg2", pinned, uniq_name = "_QFfirstprivate_realEarg2"}
!CHECK:     %[[ARG2_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG2_PVT]] {uniq_name = "_QFfirstprivate_realEarg2"} : (!fir.ref<f16>) -> (!fir.ref<f16>, !fir.ref<f16>)
!CHECK:     %[[ARG2_VAL:.*]] = fir.load %[[ARG2_DECL]]#0 : !fir.ref<f16>
!CHECK:     hlfir.assign %[[ARG2_VAL]] to %[[ARG2_PVT_DECL]]#0 temporary_lhs : f16, !fir.ref<f16>
!CHECK:     %[[ARG3_PVT:.*]] = fir.alloca f32 {bindc_name = "arg3", pinned, uniq_name = "_QFfirstprivate_realEarg3"}
!CHECK:     %[[ARG3_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG3_PVT]] {uniq_name = "_QFfirstprivate_realEarg3"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:     %[[ARG3_VAL:.*]] = fir.load %[[ARG3_DECL]]#0 : !fir.ref<f32>
!CHECK:     hlfir.assign %[[ARG3_VAL]] to %[[ARG3_PVT_DECL]]#0 temporary_lhs : f32, !fir.ref<f32>
!CHECK:     %[[ARG4_PVT:.*]] = fir.alloca f64 {bindc_name = "arg4", pinned, uniq_name = "_QFfirstprivate_realEarg4"}
!CHECK:     %[[ARG4_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG4_PVT]] {uniq_name = "_QFfirstprivate_realEarg4"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
!CHECK:     %[[ARG4_VAL:.*]] = fir.load %[[ARG4_DECL]]#0 : !fir.ref<f64>
!CHECK:     hlfir.assign %[[ARG4_VAL]] to %[[ARG4_PVT_DECL]]#0 temporary_lhs : f64, !fir.ref<f64>
!CHECK:     %[[ARG5_PVT:.*]] = fir.alloca f80 {bindc_name = "arg5", pinned, uniq_name = "_QFfirstprivate_realEarg5"}
!CHECK:     %[[ARG5_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG5_PVT]] {uniq_name = "_QFfirstprivate_realEarg5"} : (!fir.ref<f80>) -> (!fir.ref<f80>, !fir.ref<f80>)
!CHECK:     %[[ARG5_VAL:.*]] = fir.load %[[ARG5_DECL]]#0 : !fir.ref<f80>
!CHECK:     hlfir.assign %[[ARG5_VAL]] to %[[ARG5_PVT_DECL]]#0 temporary_lhs : f80, !fir.ref<f80>
!CHECK:     %[[ARG6_PVT:.*]] = fir.alloca f128 {bindc_name = "arg6", pinned, uniq_name = "_QFfirstprivate_realEarg6"}
!CHECK:     %[[ARG6_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG6_PVT]] {uniq_name = "_QFfirstprivate_realEarg6"} : (!fir.ref<f128>) -> (!fir.ref<f128>, !fir.ref<f128>)
!CHECK:     %[[ARG6_VAL:.*]] = fir.load %[[ARG6_DECL]]#0 : !fir.ref<f128>
!CHECK:     hlfir.assign %[[ARG6_VAL]] to %[[ARG6_PVT_DECL]]#0 temporary_lhs : f128, !fir.ref<f128>
!CHECK:     fir.call @_QPqux(%[[ARG1_PVT_DECL]]#1, %[[ARG2_PVT_DECL]]#1, %[[ARG3_PVT_DECL]]#1, %[[ARG4_PVT_DECL]]#1, %[[ARG5_PVT_DECL]]#1, %[[ARG6_PVT_DECL]]#1) {{.*}}: (!fir.ref<f32>, !fir.ref<f16>, !fir.ref<f32>, !fir.ref<f64>, !fir.ref<f80>, !fir.ref<f128>) -> ()
!CHECK:     omp.terminator
!CHECK:   }

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

!CHECK-LABEL:   func.func @_QPmultiple_firstprivate(
!CHECK-SAME:                                        %[[A_ADDR:.*]]: !fir.ref<i32> {fir.bindc_name = "a"},
!CHECK-SAME:                                        %[[B_ADDR:.*]]: !fir.ref<i32> {fir.bindc_name = "b"}) {
!CHECK:           %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ADDR]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFmultiple_firstprivateEa"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           %[[B_DECL:.*]]:2 = hlfir.declare %[[B_ADDR]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFmultiple_firstprivateEb"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           omp.parallel   {
!CHECK:             %[[A_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFmultiple_firstprivateEa"}
!CHECK:             %[[A_PRIV_DECL:.*]]:2 = hlfir.declare %[[A_PRIV_ADDR]] {uniq_name = "_QFmultiple_firstprivateEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:             %[[A:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
!CHECK:             hlfir.assign %[[A]] to %[[A_PRIV_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:             %[[B_PRIV_ADDR:.*]] = fir.alloca i32 {bindc_name = "b", pinned, uniq_name = "_QFmultiple_firstprivateEb"}
!CHECK:             %[[B_PRIV_DECL:.*]]:2 = hlfir.declare %[[B_PRIV_ADDR]] {uniq_name = "_QFmultiple_firstprivateEb"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:             %[[B:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<i32>
!CHECK:             hlfir.assign %[[B]] to %[[B_PRIV_DECL]]#0 temporary_lhs : i32, !fir.ref<i32>
!CHECK:             fir.call @_QPquux(%[[A_PRIV_DECL]]#1, %[[B_PRIV_DECL]]#1) {{.*}}: (!fir.ref<i32>, !fir.ref<i32>) -> ()
!CHECK:             omp.terminator
!CHECK:           }
!CHECK:           return
!CHECK:         }

subroutine multiple_firstprivate(a, b)
        integer :: a, b
!$OMP PARALLEL FIRSTPRIVATE(a) FIRSTPRIVATE(b)
        call quux(a, b)
!$OMP END PARALLEL
end subroutine multiple_firstprivate
