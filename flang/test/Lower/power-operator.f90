! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test power operation lowering

! CHECK-LABEL: pow_r4_i4
subroutine pow_r4_i4(x, y, z)
  real :: x, z
  integer :: y
  z = x ** y
  ! CHECK: call @llvm.powi.f32.i32
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
  ! CHECK: call @__fs_powk_1
end subroutine

! CHECK-LABEL: pow_r8_i4
subroutine pow_r8_i4(x, y, z)
  real(8) :: x, z
  integer :: y
  z = x ** y
  ! CHECK: call @llvm.powi.f64.i32
end subroutine

! CHECK-LABEL: pow_r8_i8
subroutine pow_r8_i8(x, y, z)
  real(8) :: x, z
  integer(8) :: y
  z = x ** y
  ! CHECK: call @__fd_powk_1
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

! CHECK-LABEL: pow_i4_i4
subroutine pow_i4_i4(x, y, z)
  integer(4) :: x, y, z
  z = x ** y
  ! CHECK: call @__mth_i_ipowi
end subroutine

! CHECK-LABEL: pow_i8_i8
subroutine pow_i8_i8(x, y, z)
  integer(8) :: x, y, z
  z = x ** y
  ! CHECK: call @__mth_i_kpowk
end subroutine

! CHECK-LABEL: pow_c4_i4
subroutine pow_c4_i4(x, y, z)
  complex :: x, z
  integer :: y
  z = x ** y
  ! CHECK: call @__fc_powi_1
end subroutine

! CHECK-LABEL: pow_c4_i8
subroutine pow_c4_i8(x, y, z)
  complex :: x, z
  integer(8) :: y
  z = x ** y
  ! CHECK: call @__fc_powk_1
end subroutine

! CHECK-LABEL: pow_c8_i4
subroutine pow_c8_i4(x, y, z)
  complex(8) :: x, z
  integer :: y
  z = x ** y
  ! CHECK: call @__fz_powi_1
end subroutine

! CHECK-LABEL: pow_c8_i8
subroutine pow_c8_i8(x, y, z)
  complex(8) :: x, z
  integer(8) :: y
  z = x ** y
  ! CHECK: call @__fz_powk_1
end subroutine
