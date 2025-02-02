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

  ! Math API

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

  ! Atomic Operations

  interface atomicadd
    attributes(device) pure integer function atomicaddi(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure real function atomicaddf(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure real(8) function atomicaddd(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real(8), intent(inout) :: address
    real(8), value :: val
    end function
    attributes(device) pure integer(8) function atomicaddl(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
  end interface 
  public :: atomicadd

  interface atomicsub
    attributes(device) pure integer function atomicsubi(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure real function atomicsubf(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure real(8) function atomicsubd(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real(8), intent(inout) :: address
    real(8), value :: val
    end function
    attributes(device) pure integer(8) function atomicsubl(address, val)
  !dir$ ignore_tkr (d) address, (dk) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
  end interface
  public :: atomicsub
  
  interface atomicmax
    attributes(device) pure integer function atomicmaxi(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure real function atomicmaxf(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure real(8) function atomicmaxd(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real(8), intent(inout) :: address
    real(8), value :: val
    end function
    attributes(device) pure integer(8) function atomicmaxl(address, val)
  !dir$ ignore_tkr (d) address, (dk) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
  end interface
  public :: atomicmax
  
  interface atomicmin
    attributes(device) pure integer function atomicmini(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
    attributes(device) pure real function atomicminf(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real, intent(inout) :: address
    real, value :: val
    end function
    attributes(device) pure real(8) function atomicmind(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    real(8), intent(inout) :: address
    real(8), value :: val
    end function
    attributes(device) pure integer(8) function atomicminl(address, val)
  !dir$ ignore_tkr (d) address, (dk) val
    integer(8), intent(inout) :: address
    integer(8), value :: val
    end function
  end interface
  public :: atomicmin
  
  interface atomicand
    attributes(device) pure integer function atomicandi(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface
  public :: atomicand
  
  interface atomicor
    attributes(device) pure integer function atomicori(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface
  public :: atomicor

  interface atomicinc
    attributes(device) pure integer function atomicinci(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface
  public :: atomicinc
  
  interface atomicdec
    attributes(device) pure integer function atomicdeci(address, val)
  !dir$ ignore_tkr (d) address, (d) val
    integer, intent(inout) :: address
    integer, value :: val
    end function
  end interface
  public :: atomicdec


end module
