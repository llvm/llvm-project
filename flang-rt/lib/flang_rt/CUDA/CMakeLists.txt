#===-- lib/flang_rt/CUDA/CMakeLists.txt ------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flangrt_library(CufRuntime STATIC SHARED
  allocatable.cpp
  allocator.cpp
  descriptor.cpp
  kernel.cpp
  memmove-function.cpp
  memory.cpp
  registration.cpp

  TARGET_PROPERTIES
    OUTPUT_NAME "CufRuntime_cuda_${CUDAToolkit_VERSION_MAJOR}"
  INCLUDE_DIRECTORIES
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

# For the static library, link-in the static dependencies as well.
if (TARGET CufRuntime.static)
  target_link_libraries(CufRuntime.static PUBLIC
    flang_rt.static
    CUDA::cudart_static
  )
endif ()

if (TARGET CufRuntime.shared)
  target_link_libraries(CufRuntime.shared PUBLIC
    flang_rt.shared
    CUDA::cudart
  )
endif ()
