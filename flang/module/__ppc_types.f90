!===-- module/__ppc_types.f90 ----------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

module __ppc_types
  private
  ! Definition of derived-types that represent PowerPC vector types.
  type __builtin_ppc_intrinsic_vector(element_category, element_kind)
    integer, kind :: element_category, element_kind
    integer(16) :: storage
  end type

  type __builtin_ppc_pair_vector
    integer(16) :: storage1
    integer(16) :: storage2
  end type

  type __builtin_ppc_quad_vector
    integer(16) :: storage1
    integer(16) :: storage2
    integer(16) :: storage3
    integer(16) :: storage4
  end type

  public :: __builtin_ppc_intrinsic_vector
  public :: __builtin_ppc_pair_vector
  public :: __builtin_ppc_quad_vector

end module __ppc_types
