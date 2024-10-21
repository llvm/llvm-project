#===-- unittests/Runtime/CUDA/CMakeLists.txt -------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flangrt_unittest(FlangCufRuntimeTests
  AllocatorCUF.cpp
)

target_link_libraries(FlangCufRuntimeTests
  PRIVATE
  CufRuntime_cuda_${CUDAToolkit_VERSION_MAJOR}
  flang_rt
)

target_include_directories(FlangCufRuntimeTests PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
