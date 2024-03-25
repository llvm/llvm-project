! This test checks lowering of OpenMP Threadprivate Directive.
! Test for variables with different kind.

!REQUIRES: shell
!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s

program test
  integer, save :: i
  integer(kind=1), save :: i1
  integer(kind=2), save :: i2
  integer(kind=4), save :: i4
  integer(kind=8), save :: i8
  integer(kind=16), save :: i16

!CHECK-DAG:  [[ADDR0:%.*]] = fir.address_of(@_QFEi) : !fir.ref<i32>
!CHECK-DAG:  [[NEWADDR0:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  [[ADDR1:%.*]] = fir.address_of(@_QFEi1) : !fir.ref<i8>
!CHECK-DAG:  [[NEWADDR1:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<i8> -> !fir.ref<i8>
!CHECK-DAG:  [[ADDR2:%.*]] = fir.address_of(@_QFEi16) : !fir.ref<i128>
!CHECK-DAG:  [[NEWADDR2:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<i128> -> !fir.ref<i128>
!CHECK-DAG:  [[ADDR3:%.*]] = fir.address_of(@_QFEi2) : !fir.ref<i16>
!CHECK-DAG:  [[NEWADDR3:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<i16> -> !fir.ref<i16>
!CHECK-DAG:  [[ADDR4:%.*]] = fir.address_of(@_QFEi4) : !fir.ref<i32>
!CHECK-DAG:  [[NEWADDR4:%.*]] = omp.threadprivate [[ADDR4]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:  [[ADDR5:%.*]] = fir.address_of(@_QFEi8) : !fir.ref<i64>
!CHECK-DAG:  [[NEWADDR5:%.*]] = omp.threadprivate [[ADDR5]] : !fir.ref<i64> -> !fir.ref<i64>
  !$omp threadprivate(i, i1, i2, i4, i8, i16)

!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<i128>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<i16>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR4]] : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR5]] : !fir.ref<i64>
  print *, i, i1, i2, i4, i8, i16

  !$omp parallel
!CHECK-DAG:    [[ADDR39:%.*]] = omp.threadprivate [[ADDR0]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:    [[ADDR40:%.*]] = omp.threadprivate [[ADDR1]] : !fir.ref<i8> -> !fir.ref<i8>
!CHECK-DAG:    [[ADDR41:%.*]] = omp.threadprivate [[ADDR2]] : !fir.ref<i128> -> !fir.ref<i128>
!CHECK-DAG:    [[ADDR42:%.*]] = omp.threadprivate [[ADDR3]] : !fir.ref<i16> -> !fir.ref<i16>
!CHECK-DAG:    [[ADDR43:%.*]] = omp.threadprivate [[ADDR4]] : !fir.ref<i32> -> !fir.ref<i32>
!CHECK-DAG:    [[ADDR44:%.*]] = omp.threadprivate [[ADDR5]] : !fir.ref<i64> -> !fir.ref<i64>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR39]] : !fir.ref<i32>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR40]] : !fir.ref<i8>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR41]] : !fir.ref<i128>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR42]] : !fir.ref<i16>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR43]] : !fir.ref<i32>
!CHECK-DAG:    %{{.*}} = fir.load [[ADDR44]] : !fir.ref<i64>
    print *, i, i1, i2, i4, i8, i16
  !$omp end parallel

!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR0]] : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR1]] : !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR2]] : !fir.ref<i128>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR3]] : !fir.ref<i16>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR4]] : !fir.ref<i32>
!CHECK-DAG:  %{{.*}} = fir.load [[NEWADDR5]] : !fir.ref<i64>
  print *, i, i1, i2, i4, i8, i16

!CHECK-DAG: fir.global internal @_QFEi : i32 {
!CHECK-DAG: fir.global internal @_QFEi1 : i8 {
!CHECK-DAG: fir.global internal @_QFEi16 : i128 {
!CHECK-DAG: fir.global internal @_QFEi2 : i16 {
!CHECK-DAG: fir.global internal @_QFEi4 : i32 {
!CHECK-DAG: fir.global internal @_QFEi8 : i64 {
end
