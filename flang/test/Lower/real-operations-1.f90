! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}
! Test real add on real kinds.

! CHECK-LABEL: real2
REAL(2) FUNCTION real2(x0, x1)
  REAL(2) :: x0
  REAL(2) :: x1
  ! CHECK: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<f16>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<f16>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: f16
  real2 = x0 + x1
  ! CHECK: return %{{.*}} : f16
END FUNCTION real2

! CHECK-LABEL: real3
REAL(3) FUNCTION real3(x0, x1)
  REAL(3) :: x0
  REAL(3) :: x1
  ! CHECK: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<bf16>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<bf16>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: bf16
  real3 = x0 + x1
  ! CHECK: return %{{.*}} : bf16
END FUNCTION real3

! CHECK-LABEL: real4
REAL(4) FUNCTION real4(x0, x1)
  REAL(4) :: x0
  REAL(4) :: x1
  ! CHECK: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<f32>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<f32>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: f32
  real4 = x0 + x1
  ! CHECK: return %{{.*}} : f32
END FUNCTION real4

! CHECK-LABEL: defreal
REAL FUNCTION defreal(x0, x1)
  REAL :: x0
  REAL :: x1
  ! CHECK: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<f32>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<f32>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: f32
  defreal = x0 + x1
  ! CHECK: return %{{.*}} : f32
END FUNCTION defreal

! CHECK-LABEL: real8
REAL(8) FUNCTION real8(x0, x1)
  REAL(8) :: x0
  REAL(8) :: x1
  ! CHECK: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<f64>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<f64>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: f64
  real8 = x0 + x1
  ! CHECK: return %{{.*}} : f64
END FUNCTION real8

! CHECK-LABEL: doubleprec
DOUBLE PRECISION FUNCTION doubleprec(x0, x1)
  DOUBLE PRECISION :: x0
  DOUBLE PRECISION :: x1
  ! CHECK: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<f64>
  ! CHECK-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<f64>
  ! CHECK: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: f64
  doubleprec = x0 + x1
  ! CHECK: return %{{.*}} : f64
END FUNCTION doubleprec

! CHECK-KIND10-LABEL: real10
FUNCTION real10(x0, x1)
  INTEGER, PARAMETER :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  REAL(kind10) :: real10
  REAL(kind10) :: x0
  REAL(kind10) :: x1
  ! CHECK-KIND10: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK-KIND10: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-KIND10-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<f80>
  ! CHECK-KIND10-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<f80>
  ! CHECK-KIND10: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: f80
  real10 = x0 + x1
  ! CHECK-KIND10: return %{{.*}} : f80
END FUNCTION real10

! CHECK-KIND16-LABEL: real16(
FUNCTION real16(x0, x1)
  INTEGER, PARAMETER :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  REAL(kind16) :: real16
  REAL(kind16) :: x0
  REAL(kind16) :: x1
  ! CHECK-KIND16: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK-KIND16: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-KIND16-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<f128>
  ! CHECK-KIND16-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<f128>
  ! CHECK-KIND16: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: f128
  real16 = x0 + x1
  ! CHECK-KIND16: return %{{.*}} : f128
END FUNCTION real16

! CHECK-KIND16-LABEL: real16b
FUNCTION real16b(x0, x1)
  INTEGER, PARAMETER :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  REAL(kind16) :: real16b
  REAL(kind16) :: x0
  REAL(kind16) :: x1
  ! CHECK-KIND16: %[[x0:.*]]:2 = hlfir.declare{{.*}}x0"
  ! CHECK-KIND16: %[[x1:.*]]:2 = hlfir.declare{{.*}}x1"
  ! CHECK-KIND16-DAG: %[[v1:.+]] = fir.load %[[x0]]#0 : !fir.ref<f128>
  ! CHECK-KIND16-DAG: %[[v2:.+]] = fir.load %[[x1]]#0 : !fir.ref<f128>
  ! CHECK-KIND16-DAG: %[[v3:.+]] = arith.addf %[[v1]], %[[v2]] {{.*}}: f128
  ! CHECK-KIND16-DAG: %[[v0:.+]] = arith.constant 4.0{{.*}} : f128
  ! CHECK-KIND16: %[[v4:.+]] = arith.subf %[[v3]], %[[v0]] {{.*}}: f128
  real16b = x0 + x1 - 4.0_16
  ! CHECK-KIND16: return %{{.*}} : f128
END FUNCTION real16b
