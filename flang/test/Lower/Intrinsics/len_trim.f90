! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: len_trim_test
integer function len_trim_test(c)
character(*) :: c
ltrim = len_trim(c)
! CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
! CHECK-DAG: %[[cm1:.*]] = arith.constant -1 : index
! CHECK-DAG: %[[lastChar:.*]] = arith.subi {{.*}}, %[[c1]]
! CHECK: %[[iterateResult:.*]]:2 = fir.iterate_while (%[[index:.*]] = %[[lastChar]] to %[[c0]] step %[[cm1]]) and ({{.*}}) iter_args({{.*}}) {
  ! CHECK: %[[addr:.*]] = fir.coordinate_of {{.*}}, %[[index]]
  ! CHECK: %[[codeAddr:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[code:.*]] = fir.load %[[codeAddr]]
  ! CHECK: %[[bool:.*]] = arith.cmpi eq
  ! CHECK: fir.result %[[bool]], %[[index]]
! CHECK: }
! CHECK: %[[len:.*]] = arith.addi %[[iterateResult]]#1, %[[c1]]
! CHECK: select %[[iterateResult]]#0, %[[c0]], %[[len]]
end function

! CHECK-LABEL: lnblnk_test
integer function lnblnk_test(c)
character(*) :: c
ltrim = lnblnk(c)
! CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
! CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
! CHECK-DAG: %[[cm1:.*]] = arith.constant -1 : index
! CHECK-DAG: %[[lastChar:.*]] = arith.subi {{.*}}, %[[c1]]
! CHECK: %[[iterateResult:.*]]:2 = fir.iterate_while (%[[index:.*]] = %[[lastChar]] to %[[c0]] step %[[cm1]]) and ({{.*}}) iter_args({{.*}}) {
  ! CHECK: %[[addr:.*]] = fir.coordinate_of {{.*}}, %[[index]]
  ! CHECK: %[[codeAddr:.*]] = fir.convert %[[addr]]
  ! CHECK: %[[code:.*]] = fir.load %[[codeAddr]]
  ! CHECK: %[[bool:.*]] = arith.cmpi eq
  ! CHECK: fir.result %[[bool]], %[[index]]
! CHECK: }
! CHECK: %[[len:.*]] = arith.addi %[[iterateResult]]#1, %[[c1]]
! CHECK: select %[[iterateResult]]#0, %[[c0]], %[[len]]
end function
