!===-- module/cudedevice.f90 -----------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! CUDA Fortran procedures available in device subprogram

module cudadevice
implicit none

  ! Set PRIVATE by default to explicitly only export what is meant
  ! to be exported by this MODULE.
  private

  ! Synchronization Functions

  interface
    attributes(device) subroutine syncthreads() bind(c, name='__syncthreads')
    end subroutine
  end interface
  public :: syncthreads

  interface
    attributes(device) integer function syncthreads_and(value) bind(c, name='__syncthreads_and')
      integer :: value
    end function
  end interface
  public :: syncthreads_and

  interface
    attributes(device) integer function syncthreads_count(value) bind(c, name='__syncthreads_count')
      integer :: value
    end function
  end interface
  public :: syncthreads_count

  interface
    attributes(device) integer function syncthreads_or(value) bind(c, name='__syncthreads_or')
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
    attributes(device) subroutine threadfence() bind(c, name='__threadfence')
    end subroutine
  end interface
  public :: threadfence

  interface
    attributes(device) subroutine threadfence_block() bind(c, name='__threadfence_block')
    end subroutine
  end interface
  public :: threadfence_block

  interface
    attributes(device) subroutine threadfence_system() bind(c, name='__threadfence_system')
    end subroutine
  end interface
  public :: threadfence_system

end module
