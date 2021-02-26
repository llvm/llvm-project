! RUN: bbc -emit-fir %s -o - -math-runtime=fast | FileCheck %s

! Test power operation lowering

! CHECK-LABEL: pow_r4_i4
subroutine pow_r4_i4(x, y, z)
  real :: x, z
  integer :: y
  z = x ** y
  ! CHECK: call @__fs_powi_1
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
  ! CHECK: call @__fd_powi_1
end subroutine

! CHECK-LABEL: pow_r8_i8
subroutine pow_r8_i8(x, y, z)
  real(8) :: x, z
  integer(8) :: y
  z = x ** y
  ! CHECK: call @__fd_powk_1
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
