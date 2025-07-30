!===-- module/cooperative_groups.f90 ---------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! CUDA Fortran cooperative groups

module cooperative_groups

use, intrinsic :: __fortran_builtins, only: c_devptr => __builtin_c_devptr

implicit none

type :: grid_group
  type(c_devptr), private :: handle
  integer(4) :: size
  integer(4) :: rank
end type grid_group

type :: coalesced_group
  type(c_devptr), private :: handle
  integer(4) :: size
  integer(4) :: rank
end type coalesced_group

type :: thread_group
  type(c_devptr), private :: handle
  integer(4) :: size
  integer(4) :: rank
end type thread_group

interface
  attributes(device) function this_grid()
    import
    type(grid_group) :: this_grid
  end function
end interface

interface
  attributes(device) function this_thread_block()
    import
    type(thread_group) :: this_thread_block
  end function
end interface

interface this_warp
  attributes(device) function this_warp()
    import
    type(coalesced_group) :: this_warp
  end function
end interface

end module
