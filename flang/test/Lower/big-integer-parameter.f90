! Test correct lowering of 128 bit integer parameters.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

program i128
  integer(16), parameter :: maxi64 = 9223372036854775807_16
  integer(16), parameter :: mini64 = -9223372036854775808_16
  integer(16), parameter :: maxi128 = 170141183460469231731687303715884105727_16
  integer(16), parameter :: mini128 = -170141183460469231731687303715884105728_16
  integer(16), parameter :: x = 9223372036854775808_16
  integer(16), parameter :: y = -9223372036854775809_16
  integer(16), parameter :: z = 0_16
  print*,x
  print*,y
end

! CHECK-LABEL: func.func @_QQmain() {
! CHECK-COUNT-2:  %{{.*}} = fir.call @_FortranAioOutputInteger128(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i128) -> i1


! CHECK-LABEL: fir.global internal @_QFECmaxi128 constant : i128 {
! CHECK-NEXT: %{{.*}} = arith.constant 170141183460469231731687303715884105727 : i128

! CHECK-LABEL: fir.global internal @_QFECmaxi64 constant : i128 {
! CHECK-NEXT:   %{{.*}} = arith.constant 9223372036854775807 : i128

! CHECK-LABEL: fir.global internal @_QFECmini128 constant : i128 {
! CHECK-NEXT: %{{.*}} = arith.constant -170141183460469231731687303715884105728 : i128

! CHECK-LABEL: fir.global internal @_QFECmini64 constant : i128 {
! CHECK-NEXT: %{{.*}} = arith.constant -9223372036854775808 : i128

! CHECK-LABEL: fir.global internal @_QFECx constant : i128 {
! CHECK-NEXT:   %{{.*}} = arith.constant 9223372036854775808 : i128
 
! CHECK-LABEL: fir.global internal @_QFECy constant : i128 {
! CHECK-NEXT: %{{.*}} = arith.constant -9223372036854775809 : i128

! CHECK-LABEL: fir.global internal @_QFECz constant : i128 {
! CHECK-NEXT: %{{.*}} = arith.constant 0 : i128
