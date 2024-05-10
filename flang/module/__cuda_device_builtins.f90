!===-- module/__cuda_device_builtins.f90 -----------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! CUDA Fortran procedures available in device subprogram

module __CUDA_device_builtins  

  implicit none

  ! Set PRIVATE by default to explicitly only export what is meant
  ! to be exported by this MODULE.
  private

  ! Synchronization Functions

  interface
    subroutine __cuda_device_builtins_syncthreads()
    end subroutine
  end interface
  public :: __cuda_device_builtins_syncthreads

  interface
    integer function __cuda_device_builtins_syncthreads_and(value)
      integer :: value
    end function
  end interface
  public :: __cuda_device_builtins_syncthreads_and

  interface
    integer function __cuda_device_builtins_syncthreads_count(value)
      integer :: value
    end function
  end interface
  public :: __cuda_device_builtins_syncthreads_count

  interface
    integer function __cuda_device_builtins_syncthreads_or(int_value)
    end function
  end interface
  public :: __cuda_device_builtins_syncthreads_or

  interface
    subroutine __cuda_device_builtins_syncwarp(mask)
      integer :: mask
    end subroutine
  end interface
  public :: __cuda_device_builtins_syncwarp

  ! Memory Fences

  interface
    subroutine __cuda_device_builtins_threadfence()
    end subroutine
  end interface
  public :: __cuda_device_builtins_threadfence

  interface
    subroutine __cuda_device_builtins_threadfence_block()
    end subroutine
  end interface
  public :: __cuda_device_builtins_threadfence_block

  interface
    subroutine __cuda_device_builtins_threadfence_system()
    end subroutine
  end interface
  public :: __cuda_device_builtins_threadfence_system

end module
