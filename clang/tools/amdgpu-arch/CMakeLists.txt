# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# // Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# // Notified per clause 4(b) of the license.
# //
# //===----------------------------------------------------------------------===//

find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)

set(AMDGPU_ARCH_DLOPEN_LIBHSA OFF)
option(AMDGPU_ARCH_FORCE_DLOPEN_LIBHSA "Build with dlopened libhsa" ${AMDGPU_ARCH_DLOPEN_LIBHSA})

if (${hsa-runtime64_FOUND} AND NOT AMDGPU_ARCH_FORCE_DLOPEN_LIBHSA)
  message(STATUS "Building amdgpu-arch linked against libhsa")
  add_clang_tool(amdgpu-arch AMDGPUArch.cpp)
  set(LINK_LIBS hsa-runtime64::hsa-runtime64)
elseif(UNIX)
  message(STATUS "Building amdgpu-arch for dlopened libhsa")
  include_directories(dynamic_hsa)
  add_clang_tool(amdgpu-arch AMDGPUArch.cpp dynamic_hsa/hsa.cpp)
  set(LINK_LIBS ${CMAKE_DL_LIBS})
else()
  message(STATUS "Not building amdgpu-arch for dlopened libhsa on non-UNIX systems.")
  return()
endif()

set_target_properties(amdgpu-arch PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
target_link_libraries(amdgpu-arch PRIVATE ${LINK_LIBS})
