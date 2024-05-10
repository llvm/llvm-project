!===-- module/cudedevice.f90 -----------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! CUDA Fortran procedures available in device subprogram

module cudadevice
  use __cuda_device_builtins, only: &
    syncthreads => __cuda_device_builtins_syncthreads, &
    syncthreads_and => __cuda_device_builtins_syncthreads_and, &
    syncthreads_count => __cuda_device_builtins_syncthreads_count, &
    syncthreads_or => __cuda_device_builtins_syncthreads_or, &
    syncwarp => __cuda_device_builtins_syncwarp, &
    threadfence => __cuda_device_builtins_threadfence, &
    threadfence_block => __cuda_device_builtins_threadfence_block, &
    threadfence_system => __cuda_device_builtins_threadfence_system
end module
