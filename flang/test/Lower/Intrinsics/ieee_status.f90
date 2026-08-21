! Tests HLFIR code generation for ieee_get_status / ieee_set_status on PPC targets.
!
! REQUIRES: powerpc-registered-target
! RUN: %flang_fc1 -triple powerpc64-ibm-aix -emit-hlfir -o - %s | FileCheck %s --check-prefix=CHECK-AIX
! RUN: %flang_fc1 -triple powerpc64le-unknown-linux-gnu -emit-hlfir -o - %s | FileCheck %s --check-prefix=CHECK-LNX

program test
  use ieee_arithmetic
  type(ieee_status_type) :: stat

! CHECK-AIX-LABEL: func.func @_QQmain
! CHECK-LNX-LABEL: func.func @_QQmain

  call ieee_get_status(stat)

! CHECK-AIX: %[[UUDAT:.*]] = fir.convert {{.*}} : ({{.*}}) -> !fir.ref<!fir.array<?xi8>>
! CHECK-AIX: %[[C1:.*]] = arith.constant 0 : index
! CHECK-AIX: %[[UUDAT0:.*]] = fir.coordinate_of %[[UUDAT]], %[[C1]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK-AIX: %[[FENV:.*]] = fir.convert %[[UUDAT0]] : (!fir.ref<i8>) -> !fir.ref<i32>
! CHECK-AIX: %[[C2:.*]] = arith.constant 20 : index
! CHECK-AIX: %[[UUDAT1:.*]] = fir.coordinate_of %[[UUDAT]], %[[C2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK-AIX: %[[UUDAT1F:.*]] = fir.convert %[[UUDAT1]] : (!fir.ref<i8>) -> !fir.ref<f64>
! CHECK-AIX: {{.*}} = fir.call @fegetenv(%[[FENV]]) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK-AIX: %[[FPS:.*]] = fir.call @llvm.ppc.readflm() {{.*}} : () -> f64
! CHECK-AIX: fir.store %[[FPS]] to %[[UUDAT1F]] : !fir.ref<f64>

! CHECK-LNX: %[[UUDAT:.*]] = fir.convert {{.*}} : ({{.*}}) -> !fir.ref<i32>
! CHECK-LNX: {{.*}} = fir.call @fegetenv(%[[UUDAT]]) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK-LNX-NOT: @llvm.ppc.readflm()

  call ieee_set_status(stat)

! CHECK-AIX: %[[UUDAT:.*]] = fir.convert {{.*}} : ({{.*}}) -> !fir.ref<!fir.array<?xi8>>
! CHECK-AIX: %[[C3:.*]] = arith.constant 0 : index
! CHECK-AIX: %[[UUDAT0:.*]] = fir.coordinate_of %[[UUDAT]], %[[C3]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK-AIX: %[[FENV:.*]] = fir.convert %[[UUDAT0]] : (!fir.ref<i8>) -> !fir.ref<i32>
! CHECK-AIX: %[[C4:.*]] = arith.constant 20 : index
! CHECK-AIX: %[[UUDAT1:.*]] = fir.coordinate_of %[[UUDAT]], %[[C4]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! CHECK-AIX: %[[UUDAT1F:.*]] = fir.convert %[[UUDAT1]] : (!fir.ref<i8>) -> !fir.ref<f64>
! CHECK-AIX: {{.*}} = fir.call @fesetenv(%[[FENV]]) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK-AIX: %[[FPS:.*]] = fir.load %[[UUDAT1F]] : !fir.ref<f64>
! CHECK-AIX: fir.call @llvm.ppc.setflm(%[[FPS]]) {{.*}} : (f64) -> f64

! CHECK-LNX: %[[UUDAT:.*]] = fir.convert {{.*}} : ({{.*}}) -> !fir.ref<i32>
! CHECK-LNX: {{.*}} = fir.call @fesetenv(%[[UUDAT]]) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK-LNX-NOT: @llvm.ppc.setflm()
end program
