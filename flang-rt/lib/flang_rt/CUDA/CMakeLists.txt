#===-- lib/flang_rt/CUDA/CMakeLists.txt ------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#


add_flangrt_library(CufRuntime STATIC
  allocatable.cpp
  allocator.cpp
  descriptor.cpp
  kernel.cpp
  memmove-function.cpp
  memory.cpp
  registration.cpp

  # libCufRuntime depends on a certain version of CUDA. To be able to have
  # multiple build of this library with different CUDA version, the version is
  # added to the library name.
  TARGET_PROPERTIES
    OUTPUT_NAME "CufRuntime_cuda_${CUDAToolkit_VERSION_MAJOR}"

  INCLUDE_DIRECTORIES
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(CufRuntime
  PUBLIC
  flang_rt
  CUDA::cudart_static
)
