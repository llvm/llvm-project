! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPmod_testr4(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f32>{{.*}}, %[[arg1:.*]]: !fir.ref<f32>{{.*}}, %[[arg2:.*]]: !fir.ref<f32>{{.*}}) {
subroutine mod_testr4(r, a, p)
  real(4) :: r, a, p
! CHECK: %[[V1:.*]] = fir.load %[[arg1]] : !fir.ref<f32>
! CHECK: %[[V2:.*]] = fir.load %[[arg2]] : !fir.ref<f32>
! CHECK: %[[FILE:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK: %[[LINE:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK: %[[FILEARG:.*]] = fir.convert %[[FILE]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAModReal4(%[[V1]], %[[V2]], %[[FILEARG]], %[[LINE]]) : (f32, f32, !fir.ref<i8>, i32) -> f32
  r = mod(a, p)
end subroutine

! CHECK-LABEL: func @_QPmod_testr8(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f64>{{.*}}, %[[arg1:.*]]: !fir.ref<f64>{{.*}}, %[[arg2:.*]]: !fir.ref<f64>{{.*}}) {
subroutine mod_testr8(r, a, p)
  real(8) :: r, a, p
! CHECK: %[[V1:.*]] = fir.load %[[arg1]] : !fir.ref<f64>
! CHECK: %[[V2:.*]] = fir.load %[[arg2]] : !fir.ref<f64>
! CHECK: %[[FILE:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK: %[[LINE:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK: %[[FILEARG:.*]] = fir.convert %[[FILE]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAModReal8(%[[V1]], %[[V2]], %[[FILEARG]], %[[LINE]]) : (f64, f64, !fir.ref<i8>, i32) -> f64
  r = mod(a, p)
end subroutine

! CHECK-LABEL: func @_QPmod_testr10(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f80>{{.*}}, %[[arg1:.*]]: !fir.ref<f80>{{.*}}, %[[arg2:.*]]: !fir.ref<f80>{{.*}}) {
subroutine mod_testr10(r, a, p)
  real(10) :: r, a, p
! CHECK: %[[V1:.*]] = fir.load %[[arg1]] : !fir.ref<f80>
! CHECK: %[[V2:.*]] = fir.load %[[arg2]] : !fir.ref<f80>
! CHECK: %[[FILE:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK: %[[LINE:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK: %[[FILEARG:.*]] = fir.convert %[[FILE]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAModReal10(%[[V1]], %[[V2]], %[[FILEARG]], %[[LINE]]) : (f80, f80, !fir.ref<i8>, i32) -> f80
  r = mod(a, p)
end subroutine

! CHECK-LABEL: func @_QPmod_testr16(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f128>{{.*}}, %[[arg1:.*]]: !fir.ref<f128>{{.*}}, %[[arg2:.*]]: !fir.ref<f128>{{.*}}) {
subroutine mod_testr16(r, a, p)
  real(16) :: r, a, p
! CHECK: %[[V1:.*]] = fir.load %[[arg1]] : !fir.ref<f128>
! CHECK: %[[V2:.*]] = fir.load %[[arg2]] : !fir.ref<f128>
! CHECK: %[[FILE:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK: %[[LINE:.*]] = arith.constant {{[0-9]*}} : i32
! CHECK: %[[FILEARG:.*]] = fir.convert %[[FILE]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAModReal16(%[[V1]], %[[V2]], %[[FILEARG]], %[[LINE]]) : (f128, f128, !fir.ref<i8>, i32) -> f128
  r = mod(a, p)
end subroutine
