! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 %s -o %t.o

! PACK with scalar .TRUE. mask lowers to hlfir.reshape (not _FortranAPack).
subroutine pack_scalar_true(a, r)
  integer :: a(:, :, :)
  integer :: r(:)
  r = pack(a(:, :, 1), .true.)
end subroutine pack_scalar_true

! CHECK-LABEL: func.func @_QPpack_scalar_true
! CHECK-NOT: _FortranAPack
! CHECK: hlfir.reshape

! Static explicit-shape array (gfortran torture intrinsic_pack.f90 pattern).
subroutine pack_static_scalar_true
  integer, dimension(3, 3) :: a
  integer, dimension(9) :: r
  r = pack(a, .true.)
end subroutine pack_static_scalar_true

! CHECK-LABEL: func.func @_QPpack_static_scalar_true
! CHECK-NOT: _FortranAPack
! CHECK: hlfir.reshape

! Scalar logical variable mask still uses runtime PACK.
subroutine pack_scalar_var_mask(a, m, r)
  integer :: a(:)
  logical :: m
  integer :: r(:)
  r = pack(a, m)
end subroutine pack_scalar_var_mask

! CHECK-LABEL: func.func @_QPpack_scalar_var_mask
! CHECK: fir.call @_FortranAPack

! Array mask uses the runtime PACK path.
subroutine pack_array_mask(a, m, r)
  integer :: a(:)
  logical :: m(:)
  integer :: r(:)
  r = pack(a, m)
end subroutine pack_array_mask

! CHECK-LABEL: func.func @_QPpack_array_mask
! CHECK: fir.call @_FortranAPack
