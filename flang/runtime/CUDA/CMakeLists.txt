#===-- runtime/CUDA/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

include_directories(${CUDAToolkit_INCLUDE_DIRS})

# libCufRuntime depends on a certain version of CUDA. To be able to have
# multiple build of this library with different CUDA version, the version is
# added to the library name.
set(CUFRT_LIBNAME CufRuntime_cuda_${CUDAToolkit_VERSION_MAJOR})

add_flang_library(${CUFRT_LIBNAME}
  allocator.cpp
  allocatable.cpp
  descriptor.cpp
  init.cpp
  kernel.cpp
  memmove-function.cpp
  memory.cpp
  pointer.cpp
  registration.cpp
)

if (BUILD_SHARED_LIBS)
  set(CUDA_RT_TARGET CUDA::cudart)
else()
  set(CUDA_RT_TARGET CUDA::cudart_static)
endif()

target_link_libraries(${CUFRT_LIBNAME}
  PRIVATE
  FortranRuntime
  ${CUDA_RT_TARGET}
)
