# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===----------------------------------------------------------------------===//

find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if (NOT ${hsa-runtime64_FOUND})
  message(STATUS "Not building amdgpu-arch: hsa-runtime64 not found")
  return()
endif()

add_clang_tool(amdgpu-arch AMDGPUArch.cpp)

set_target_properties(amdgpu-arch PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

clang_target_link_libraries(amdgpu-arch PRIVATE hsa-runtime64::hsa-runtime64)
