# //===--------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===--------------------------------------------------------------------===//


# TODO: This is deprecated. Since CMake 3.17 we can use FindCUDAToolkit instead.
find_package(CUDA QUIET)
find_library(cuda-library NAMES cuda PATHS /lib64)
if (NOT cuda-library AND CUDA_FOUND)
  get_filename_component(CUDA_LIBDIR "${CUDA_cudart_static_LIBRARY}" DIRECTORY)
  find_library(cuda-library NAMES cuda HINTS "${CUDA_LIBDIR}/stubs")
endif()

if (NOT CUDA_FOUND OR NOT cuda-library)
  message(STATUS "Not building nvptx-arch: cuda runtime not found")
  return()
endif()

add_clang_tool(nvptx-arch NVPTXArch.cpp)

set_target_properties(nvptx-arch PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
target_include_directories(nvptx-arch PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(nvptx-arch PRIVATE ${cuda-library})
