!===-- module/cudedevice.f90 -----------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! CUDA Fortran procedures available in device subprogram

module cudadevice
  use __cuda_device
  use, intrinsic :: __fortran_builtins, only: dim3 => __builtin_dim3
  use, intrinsic :: __fortran_builtins, only: c_devptr => __builtin_c_devptr
  use, intrinsic :: __fortran_builtins, only: c_devloc => __builtin_c_devloc
implicit none

  ! Set PRIVATE by default to explicitly only export what is meant
  ! to be exported by this MODULE.
  private

  ! Synchronization Functions

  interface
    attributes(device) subroutine syncthreads()
    end subroutine
  end interface
  public :: syncthreads

  interface
    attributes(device) integer function syncthreads_and(value)
      integer :: value
    end function
  end interface
  public :: syncthreads_and

  interface
    attributes(device) integer function syncthreads_count(value)
      integer :: value
    end function
  end interface
  public :: syncthreads_count

  interface
    attributes(device) integer function syncthreads_or(value)
      integer :: value
    end function
  end interface
  public :: syncthreads_or

  interface
    attributes(device) subroutine syncwarp(mask) bind(c, name='__syncwarp')
      integer :: mask
    end subroutine
  end interface
  public :: syncwarp

  ! Memory Fences

  interface
    attributes(device) subroutine threadfence()
    end subroutine
  end interface
  public :: threadfence

  interface
    attributes(device) subroutine threadfence_block()
    end subroutine
  end interface
  public :: threadfence_block

  interface
    attributes(device) subroutine threadfence_system()
    end subroutine
  end interface
  public :: threadfence_system

end module
