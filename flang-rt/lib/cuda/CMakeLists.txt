#===-- lib/cuda/CMakeLists.txt ---------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

add_flangrt_library(flang_rt.cuda STATIC SHARED
  allocatable.cpp
  allocator.cpp
  descriptor.cpp
  init.cpp
  kernel.cpp
  memmove-function.cpp
  memory.cpp
  registration.cpp

  TARGET_PROPERTIES
    # libflang_rt.runtime depends on a certain version of CUDA. To be able to have
    # multiple build of this library with different CUDA version, the version is
    # added to the library name.
    OUTPUT_NAME "flang_rt.cuda_${CUDAToolkit_VERSION_MAJOR}"
  INCLUDE_DIRECTORIES
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
  INSTALL_WITH_TOOLCHAIN
)

# For the static library, link-in the static dependencies as well.
if (TARGET flang_rt.cuda.static)
  target_link_libraries(flang_rt.cuda.static PUBLIC
    flang_rt.runtime.static
    CUDA::cudart_static
  )
endif ()

# For the shared library, use the shared versions of the dependencies.
if (TARGET flang_rt.cuda.shared)
  target_link_libraries(flang_rt.cuda.shared PUBLIC
    flang_rt.runtime.shared
    CUDA::cudart
  )
endif ()
