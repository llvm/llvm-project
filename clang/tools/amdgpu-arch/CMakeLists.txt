# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===----------------------------------------------------------------------===//

set(LLVM_LINK_COMPONENTS Support)

add_clang_tool(amdgpu-arch AMDGPUArch.cpp)

# If we find the HSA runtime we link with it directly.
find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(hsa-runtime64_FOUND AND NOT (LLVM_BUILD_32_BITS OR CMAKE_SIZEOF_VOID_P EQUAL 4))
  set_target_properties(amdgpu-arch PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
  clang_target_link_libraries(amdgpu-arch PRIVATE hsa-runtime64::hsa-runtime64)
else()
  target_compile_definitions(amdgpu-arch PRIVATE "DYNAMIC_HSA")
endif()
