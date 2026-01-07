! This test checks lowering of OpenMP Threadprivate Directive.
! Test for variables with different kind.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program test
  integer, save :: i
  integer(kind=1), save :: i1
  integer(kind=2), save :: i2
  integer(kind=4), save :: i4
  integer(kind=8), save :: i8
  integer(kind=16), save :: i16

!CHECK-DAG:  %[[I:.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
!CHECK-DAG:  %[[I_DECL:.*]]:2 = hlfir.declare %[[I]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:  %[[OMP_I:.*]] = omp.threadprivate %[[I_DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  %[[OMP_I_DECL:.*]]:2 = hlfir.declare %[[OMP_I]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:  %[[I1:.*]] = fir.address_of(@_QFEi1) : !fir.ref<i8>
!CHECK-DAG:  %[[I1_DECL:.*]]:2 = hlfir.declare %[[I1]] {uniq_name = "_QFEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK-DAG:  %[[OMP_I1:.*]] = omp.threadprivate %[[I1_DECL]]#0 : !fir.ref<i8> -> !fir.ref<i8>
!CHECK-DAG:  %[[OMP_I1_DECL:.*]]:2 = hlfir.declare %[[OMP_I1]] {uniq_name = "_QFEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK-DAG:  %[[I16:.*]] = fir.address_of(@_QFEi16) : !fir.ref<i128>
!CHECK-DAG:  %[[I16_DECL:.*]]:2 = hlfir.declare %[[I16]] {uniq_name = "_QFEi16"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
!CHECK-DAG:  %[[OMP_I16:.*]] = omp.threadprivate %[[I16_DECL]]#0 : !fir.ref<i128> -> !fir.ref<i128>
!CHECK-DAG:  %[[OMP_I16_DECL:.*]]:2 = hlfir.declare %[[OMP_I16]] {uniq_name = "_QFEi16"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
!CHECK-DAG:  %[[I2:.*]] = fir.address_of(@_QFEi2) : !fir.ref<i16>
!CHECK-DAG:  %[[I2_DECL:.*]]:2 = hlfir.declare %[[I2]] {uniq_name = "_QFEi2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!CHECK-DAG:  %[[OMP_I2:.*]] = omp.threadprivate %[[I2_DECL]]#0 : !fir.ref<i16> -> !fir.ref<i16>
!CHECK-DAG:  %[[OMP_I2_DECL:.*]]:2 = hlfir.declare %[[OMP_I2]] {uniq_name = "_QFEi2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!CHECK-DAG:  %[[I4:.*]] = fir.address_of(@_QFEi4) : !fir.ref<i32>
!CHECK-DAG:  %[[I4_DECL:.*]]:2 = hlfir.declare %[[I4]] {uniq_name = "_QFEi4"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:  %[[OMP_I4:.*]] = omp.threadprivate %[[I4_DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  %[[OMP_I4_DECL:.*]]:2 = hlfir.declare %[[OMP_I4]] {uniq_name = "_QFEi4"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:  %[[I8:.*]] = fir.address_of(@_QFEi8) : !fir.ref<i64>
!CHECK-DAG:  %[[I8_DECL:.*]]:2 = hlfir.declare %[[I8]] {uniq_name = "_QFEi8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!CHECK-DAG:  %[[OMP_I8:.*]] = omp.threadprivate %[[I8_DECL]]#0 : !fir.ref<i64> -> !fir.ref<i64>
!CHECK-DAG:  %[[OMP_I8_DECL:.*]]:2 = hlfir.declare %[[OMP_I8]] {uniq_name = "_QFEi8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  !$omp threadprivate(i, i1, i2, i4, i8, i16)

!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I1_DECL]]#0 : !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I16_DECL]]#0 : !fir.ref<i128>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I2_DECL]]#0 : !fir.ref<i16>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I4_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I8_DECL]]#0 : !fir.ref<i64>
  print *, i, i1, i2, i4, i8, i16

  !$omp parallel
!CHECK-DAG:  %[[I_PVT:.*]] = omp.threadprivate %[[I_DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  %[[I_PVT_DECL:.*]]:2 = hlfir.declare %[[I_PVT]] {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:  %[[I1_PVT:.*]] = omp.threadprivate %[[I1_DECL]]#0 : !fir.ref<i8> -> !fir.ref<i8>
!CHECK-DAG:  %[[I1_PVT_DECL:.*]]:2 = hlfir.declare %[[I1_PVT]] {uniq_name = "_QFEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!CHECK-DAG:  %[[I2_PVT:.*]] = omp.threadprivate %[[I2_DECL]]#0 : !fir.ref<i16> -> !fir.ref<i16>
!CHECK-DAG:  %[[I2_PVT_DECL:.*]]:2 = hlfir.declare %[[I2_PVT]] {uniq_name = "_QFEi2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!CHECK-DAG:  %[[I4_PVT:.*]] = omp.threadprivate %[[I4_DECL]]#0 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  %[[I4_PVT_DECL:.*]]:2 = hlfir.declare %[[I4_PVT]] {uniq_name = "_QFEi4"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG:  %[[I8_PVT:.*]] = omp.threadprivate %[[I8_DECL]]#0 : !fir.ref<i64> -> !fir.ref<i64>
!CHECK-DAG:  %[[I8_PVT_DECL:.*]]:2 = hlfir.declare %[[I8_PVT]] {uniq_name = "_QFEi8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!CHECK-DAG:  %[[I16_PVT:.*]] = omp.threadprivate %[[I16_DECL]]#0 : !fir.ref<i128> -> !fir.ref<i128>
!CHECK-DAG:  %[[I16_PVT_DECL:.*]]:2 = hlfir.declare %[[I16_PVT]] {uniq_name = "_QFEi16"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
!CHECK-DAG:  %{{.*}} = fir.load %[[I_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[I1_PVT_DECL]]#0 : !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.load %[[I16_PVT_DECL]]#0 : !fir.ref<i128>
!CHECK-DAG:  %{{.*}} = fir.load %[[I2_PVT_DECL]]#0 : !fir.ref<i16>
!CHECK-DAG:  %{{.*}} = fir.load %[[I4_PVT_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[I8_PVT_DECL]]#0 : !fir.ref<i64>
    print *, i, i1, i2, i4, i8, i16
  !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I1_DECL]]#0 : !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I16_DECL]]#0 : !fir.ref<i128>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I2_DECL]]#0 : !fir.ref<i16>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I4_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load %[[OMP_I8_DECL]]#0 : !fir.ref<i64>
  print *, i, i1, i2, i4, i8, i16

!CHECK-DAG: fir.global internal @_QFEi : i32 {
!CHECK-DAG: fir.global internal @_QFEi1 : i8 {
!CHECK-DAG: fir.global internal @_QFEi16 : i128 {
!CHECK-DAG: fir.global internal @_QFEi2 : i16 {
!CHECK-DAG: fir.global internal @_QFEi4 : i32 {
!CHECK-DAG: fir.global internal @_QFEi8 : i64 {
end
