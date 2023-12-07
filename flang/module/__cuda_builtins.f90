!===-- module/__cuda_builtins.f90 ------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! These CUDA predefined variables are automatically available in device
! subprograms.

module __CUDA_builtins
  use __Fortran_builtins, only: &
    threadIdx => __builtin_threadIdx, &
    blockDim => __builtin_blockDim, &
    blockIdx => __builtin_blockIdx, &
    gridDim => __builtin_gridDim, &
    warpsize => __builtin_warpsize
end module
