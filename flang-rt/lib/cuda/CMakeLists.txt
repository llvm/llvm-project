#===-- lib/cuda/CMakeLists.txt ---------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#


add_flangrt_library(flang_rt.cuda STATIC
  allocatable.cpp
  allocator.cpp
  descriptor.cpp
  init.cpp
  kernel.cpp
  memmove-function.cpp
  memory.cpp
  registration.cpp

  # libflang_rt.runtime depends on a certain version of CUDA. To be able to have
  # multiple build of this library with different CUDA version, the version is
  # added to the library name.
  TARGET_PROPERTIES
    OUTPUT_NAME "flang_rt.cuda_${CUDAToolkit_VERSION_MAJOR}"

  INCLUDE_DIRECTORIES
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(flang_rt.cuda
  PUBLIC
  flang_rt.runtime
  CUDA::cudart_static
)
