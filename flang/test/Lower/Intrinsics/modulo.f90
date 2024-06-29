! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s -check-prefixes=HONORINF,ALL
! RUN: flang-new -fc1 -menable-no-infs -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s -check-prefixes=CHECK,ALL

! ALL-LABEL: func @_QPmodulo_testr(
! ALL-SAME: %[[arg0:.*]]: !fir.ref<f64>{{.*}}, %[[arg1:.*]]: !fir.ref<f64>{{.*}}, %[[arg2:.*]]: !fir.ref<f64>{{.*}}) {
subroutine modulo_testr(r, a, p)
  real(8) :: r, a, p
  ! ALL-DAG: %[[a:.*]] = fir.load %[[arg1]] : !fir.ref<f64>
  ! ALL-DAG: %[[p:.*]] = fir.load %[[arg2]] : !fir.ref<f64>
  ! HONORINF: %[[res:.*]] = fir.call @_FortranAModuloReal8(%[[a]], %[[p]]
  ! CHECK-DAG: %[[rem:.*]] = arith.remf %[[a]], %[[p]] {{.*}}: f64
  ! CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
  ! CHECK-DAG: %[[remNotZero:.*]] = arith.cmpf une, %[[rem]], %[[zero]] {{.*}} : f64
  ! CHECK-DAG: %[[aNeg:.*]] = arith.cmpf olt, %[[a]], %[[zero]] {{.*}} : f64
  ! CHECK-DAG: %[[pNeg:.*]] = arith.cmpf olt, %[[p]], %[[zero]] {{.*}} : f64
  ! CHECK-DAG: %[[signDifferent:.*]] = arith.xori %[[aNeg]], %[[pNeg]] : i1
  ! CHECK-DAG: %[[mustAddP:.*]] = arith.andi %[[remNotZero]], %[[signDifferent]] : i1
  ! CHECK-DAG: %[[remPlusP:.*]] = arith.addf %[[rem]], %[[p]] {{.*}}: f64
  ! CHECK: %[[res:.*]] = arith.select %[[mustAddP]], %[[remPlusP]], %[[rem]] : f64
  ! ALL: fir.store %[[res]] to %[[arg0]] : !fir.ref<f64>
  r = modulo(a, p)
end subroutine
  
! ALL-LABEL: func @_QPmodulo_testi(
! ALL-SAME: %[[arg0:.*]]: !fir.ref<i64>{{.*}}, %[[arg1:.*]]: !fir.ref<i64>{{.*}}, %[[arg2:.*]]: !fir.ref<i64>{{.*}}) {
subroutine modulo_testi(r, a, p)
  integer(8) :: r, a, p
  ! CHECK-DAG: %[[a:.*]] = fir.load %[[arg1]] : !fir.ref<i64>
  ! CHECK-DAG: %[[p:.*]] = fir.load %[[arg2]] : !fir.ref<i64>
  ! CHECK-DAG: %[[rem:.*]] = arith.remsi %[[a]], %[[p]] : i64
  ! CHECK-DAG: %[[argXor:.*]] = arith.xori %[[a]], %[[p]] : i64
  ! CHECK-DAG: %[[signDifferent:.*]] = arith.cmpi slt, %[[argXor]], %c0{{.*}} : i64
  ! CHECK-DAG: %[[remNotZero:.*]] = arith.cmpi ne, %[[rem]], %c0{{.*}} : i64
  ! CHECK-DAG: %[[mustAddP:.*]] = arith.andi %[[remNotZero]], %[[signDifferent]] : i1
  ! CHECK-DAG: %[[remPlusP:.*]] = arith.addi %[[rem]], %[[p]] : i64
  ! CHECK: %[[res:.*]] = arith.select %[[mustAddP]], %[[remPlusP]], %[[rem]] : i64
  ! CHECK: fir.store %[[res]] to %[[arg0]] : !fir.ref<i64>
  r = modulo(a, p)
end subroutine

! CHECK-LABEL: func @_QPmodulo_testr16(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f128>{{.*}}, %[[arg1:.*]]: !fir.ref<f128>{{.*}}, %[[arg2:.*]]: !fir.ref<f128>{{.*}}) {
subroutine modulo_testr16(r, a, p)
  real(16) :: r, a, p
  ! CHECK: fir.call @_FortranAModuloReal16({{.*}}){{.*}}: (f128, f128, !fir.ref<i8>, i32) -> f128
  r = modulo(a, p)
end subroutine
