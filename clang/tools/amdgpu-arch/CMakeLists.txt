# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===----------------------------------------------------------------------===//

set(LLVM_LINK_COMPONENTS Support)

find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)

add_clang_tool(amdgpu-arch AMDGPUArch.cpp AMDGPUArchByHSA.cpp AMDGPUArchByHIP.cpp)

target_link_libraries(amdgpu-arch PRIVATE clangBasic)
