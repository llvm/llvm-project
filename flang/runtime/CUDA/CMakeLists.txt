#===-- runtime/CUDA/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_flang_library(CufRuntime
  allocator.cpp
  descriptor.cpp
)

if (BUILD_SHARED_LIBS)
  set(CUDA_RT_TARGET CUDA::cudart)
else()
  set(CUDA_RT_TARGET CUDA::cudart_static)
endif()

target_link_libraries(CufRuntime
  PRIVATE
  FortranRuntime
  ${CUDA_RT_TARGET}
)
