! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK,PRECISE"
! RUN: bbc --math-runtime=precise -emit-fir %s -o - | FileCheck %s --check-prefixes="PRECISE"
! RUN: bbc --force-mlir-complex -emit-fir %s -o - | FileCheck %s --check-prefixes="FAST"
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK,PRECISE"
! RUN: %flang_fc1 -fapprox-func -emit-fir %s -o - | FileCheck %s --check-prefixes="CHECK,FAST"
! RUN: %flang_fc1 -emit-fir -mllvm --math-runtime=precise %s -o - | FileCheck %s --check-prefixes="PRECISE"
! RUN: %flang_fc1 -emit-fir -mllvm --force-mlir-complex %s -o - | FileCheck %s --check-prefixes="FAST"

! Test power operation lowering

! CHECK-LABEL: pow_r4_i4
subroutine pow_r4_i4(x, y, z)
  real :: x, z
  integer :: y
  z = x ** y
  ! CHECK: math.fpowi {{.*}} : f32, i32
end subroutine

! CHECK-LABEL: pow_r4_r4
subroutine pow_r4_r4(x, y, z)
  real :: x, z, y
  z = x ** y
  ! CHECK: math.powf %{{.*}}, %{{.*}} : f32
end subroutine

! CHECK-LABEL: pow_r4_i8
subroutine pow_r4_i8(x, y, z)
  real :: x, z
  integer(8) :: y
  z = x ** y
  ! CHECK: math.fpowi {{.*}} : f32, i64
end subroutine

! CHECK-LABEL: pow_r8_i4
subroutine pow_r8_i4(x, y, z)
  real(8) :: x, z
  integer :: y
  z = x ** y
  ! CHECK: math.fpowi {{.*}} : f64, i32
end subroutine

! CHECK-LABEL: pow_r8_i8
subroutine pow_r8_i8(x, y, z)
  real(8) :: x, z
  integer(8) :: y
  z = x ** y
  ! CHECK: math.fpowi {{.*}} : f64, i64
end subroutine

! CHECK-LABEL: pow_r8_r8
subroutine pow_r8_r8(x, y, z)
  real(8) :: x, z, y
  z = x ** y
  ! CHECK: math.powf %{{.*}}, %{{.*}} : f64
end subroutine

! CHECK-LABEL: pow_r4_r8
subroutine pow_r4_r8(x, y, z)
  real(4) :: x
  real(8) :: z, y
  z = x ** y
  ! CHECK: %{{.*}} = fir.convert %{{.*}} : (f32) -> f64
  ! CHECK: math.powf %{{.*}}, %{{.*}} : f64
end subroutine

! CHECK-LABEL: pow_i1_i1
subroutine pow_i1_i1(x, y, z)
  integer(1) :: x, y, z
  z = x ** y
  ! CHECK: math.ipowi %{{.*}}, %{{.*}} : i8
end subroutine

! CHECK-LABEL: pow_i2_i2
subroutine pow_i2_i2(x, y, z)
  integer(2) :: x, y, z
  z = x ** y
  ! CHECK: math.ipowi %{{.*}}, %{{.*}} : i16
end subroutine

! CHECK-LABEL: pow_i4_i4
subroutine pow_i4_i4(x, y, z)
  integer(4) :: x, y, z
  z = x ** y
  ! CHECK: math.ipowi %{{.*}}, %{{.*}} : i32
end subroutine

! CHECK-LABEL: pow_i8_i8
subroutine pow_i8_i8(x, y, z)
  integer(8) :: x, y, z
  z = x ** y
  ! CHECK: math.ipowi %{{.*}}, %{{.*}} : i64
end subroutine

! CHECK-LABEL: pow_c4_i4
subroutine pow_c4_i4(x, y, z)
  complex :: x, z
  integer :: y
  z = x ** y
  ! CHECK: call @_FortranAcpowi
end subroutine

! CHECK-LABEL: pow_c4_i8
subroutine pow_c4_i8(x, y, z)
  complex :: x, z
  integer(8) :: y
  z = x ** y
  ! CHECK: call @_FortranAcpowk
end subroutine

! CHECK-LABEL: pow_c8_i4
subroutine pow_c8_i4(x, y, z)
  complex(8) :: x, z
  integer :: y
  z = x ** y
  ! CHECK: call @_FortranAzpowi
end subroutine

! CHECK-LABEL: pow_c8_i8
subroutine pow_c8_i8(x, y, z)
  complex(8) :: x, z
  integer(8) :: y
  z = x ** y
  ! CHECK: call @_FortranAzpowk
end subroutine

! CHECK-LABEL: pow_c4_c4
subroutine pow_c4_c4(x, y, z)
  complex :: x, y, z
  z = x ** y
  ! FAST: complex.pow %{{.*}}, %{{.*}} : complex<f32>
  ! PRECISE: call @cpowf
end subroutine

! CHECK-LABEL: pow_c8_c8
subroutine pow_c8_c8(x, y, z)
  complex(8) :: x, y, z
  z = x ** y
  ! FAST: complex.pow %{{.*}}, %{{.*}} : complex<f64>
  ! PRECISE: call @cpow
end subroutine

