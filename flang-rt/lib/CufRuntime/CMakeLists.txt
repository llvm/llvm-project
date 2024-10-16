#===-- lib/CufRuntime/CMakeLists.txt ---------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# libCufRuntime depends on a certain version of CUDA. To be able to have
# multiple build of this library with different CUDA version, the version is
# added to the library name.
set(CUFRT_LIBNAME CufRuntime_cuda_${CUDAToolkit_VERSION_MAJOR})

add_flangrt_library(${CUFRT_LIBNAME}
  allocator.cpp
  descriptor.cpp
)
target_include_directories(${CUFRT_LIBNAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

if (BUILD_SHARED_LIBS)
  set(CUDA_RT_TARGET CUDA::cudart)
else()
  set(CUDA_RT_TARGET CUDA::cudart_static)
endif()

target_link_libraries(${CUFRT_LIBNAME}
  PRIVATE
  flang_rt
  ${CUDA_RT_TARGET}
)
