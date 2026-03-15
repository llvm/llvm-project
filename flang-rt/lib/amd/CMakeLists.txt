#===-- lib/amd/CMakeLists.txt ----------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# Check that Umpire exits in the directory given at CMake
# TODO: this was disabled to get to an easier build procedure for now
#message(STATUS "Using Umpire in directory ${FLANG_RT_UMPIRE_DIR}")
#set(umpire_DIR ${FLANG_RT_UMPIRE_DIR})
#find_package(umpire REQUIRED PATHS ${FLANG_RT_UMPIRE_DIR}/lib/cmake/umpire)

add_flangrt_library(flang_rt.amd STATIC SHARED
  amd_alloc.cpp
  INSTALL_WITH_TOOLCHAIN
)

#if (TARGET flang_rt.amd.static)
#  target_include_directories(flang_rt.amd.static PRIVATE ${FLANG_RT_UMPIRE_DIR}/include)
#endif()
#
#if (TARGET flang_rt.amd.shared)
#  target_include_directories(flang_rt.amd.shared PRIVATE ${FLANG_RT_UMPIRE_DIR}/include)
#endif()
