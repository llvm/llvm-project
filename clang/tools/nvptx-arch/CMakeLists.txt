# //===--------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for details.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===--------------------------------------------------------------------===//

set(LLVM_LINK_COMPONENTS Support)
add_clang_tool(nvptx-arch NVPTXArch.cpp)

find_package(CUDAToolkit QUIET)

# If we found the CUDA library directly we just dynamically link against it.
if (CUDAToolkit_FOUND AND NOT LLVM_BUILD_32_BITS)
  target_link_libraries(nvptx-arch PRIVATE CUDA::cuda_driver)
else()
  target_compile_definitions(nvptx-arch PRIVATE "DYNAMIC_CUDA")
endif()
