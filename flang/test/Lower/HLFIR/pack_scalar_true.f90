! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -O2 -S %s -o - | FileCheck %s --check-prefix=OPT
! RUN: %flang_fc1 %s -o %t.o

subroutine pack_scalar_true(a, r)
  integer :: a(:, :, :)
  integer :: r(:)
  r = pack(a(:, :, 1), .true.)
end subroutine pack_scalar_true

! CHECK-LABEL: func.func @_QPpack_scalar_true
! CHECK: hlfir.pack
! CHECK-NOT: hlfir.reshape
! OPT-LABEL: pack_scalar_true_
! OPT-NOT: _FortranAPack

subroutine pack_static_scalar_true
  integer, dimension(3, 3) :: a
  integer, dimension(9) :: r
  r = pack(a, .true.)
end subroutine pack_static_scalar_true

! CHECK-LABEL: func.func @_QPpack_static_scalar_true
! CHECK: hlfir.pack
! CHECK-NOT: hlfir.reshape
! OPT-LABEL: pack_static_scalar_true_

subroutine pack_scalar_var_mask(a, m, r)
  integer :: a(:)
  logical :: m
  integer :: r(:)
  r = pack(a, m)
end subroutine pack_scalar_var_mask

! CHECK-LABEL: func.func @_QPpack_scalar_var_mask
! CHECK: hlfir.pack
! CHECK-NOT: hlfir.reshape

subroutine pack_array_mask(a, m, r)
  integer :: a(:)
  logical :: m(:)
  integer :: r(:)
  r = pack(a, m)
end subroutine pack_array_mask

! CHECK-LABEL: func.func @_QPpack_array_mask
! CHECK: hlfir.pack
! CHECK-NOT: hlfir.reshape
