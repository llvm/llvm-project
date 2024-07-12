#===-- runtime/CUDA/CMakeLists.txt -----------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

include_directories(${CUDAToolkit_INCLUDE_DIRS})
find_library(CUDA_RUNTIME_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

add_flang_library(CufRuntime
  allocator.cpp
)
target_link_libraries(CufRuntime
PRIVATE
${CUDA_RUNTIME_LIBRARY}
)
