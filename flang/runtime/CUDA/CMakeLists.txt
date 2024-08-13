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
  set(CUF_LIBRARY ${CUDA_LIBRARIES})
else()
  set(CUF_LIBRARY ${CUDA_cudart_static_LIBRARY})
endif()

target_link_libraries(CufRuntime
  PRIVATE
  FortranRuntime
  ${CUF_LIBRARY}
)
