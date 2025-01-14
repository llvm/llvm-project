!===-- module/__cuda_device.f90 --------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! This module contains CUDA Fortran interfaces used in cudadevice.f90.

module __cuda_device
implicit none

  ! Set PRIVATE by default to explicitly only export what is meant
  ! to be exported by this MODULE.

  interface
    attributes(device) function __fadd_rd(x, y) bind(c, name='__nv_fadd_rd')
      real, intent(in), value :: x, y
      real :: __fadd_rd
    end function
  end interface
  public :: __fadd_rd

  interface
    attributes(device) function __fadd_ru(x, y) bind(c, name='__nv_fadd_ru')
      real, intent(in), value :: x, y
      real :: __fadd_ru
    end function
  end interface
  public :: __fadd_ru
end module
