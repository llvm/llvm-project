#===-- runtime/CMakeLists.txt ----------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20.0)

  project(FlangRuntime C CXX)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_EXTENSIONS OFF)

  set(FLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

  set(LLVM_COMMON_CMAKE_UTILS "${FLANG_SOURCE_DIR}/../cmake")
  set(LLVM_CMAKE_UTILS "${FLANG_SOURCE_DIR}/../llvm/cmake")
  set(CLANG_CMAKE_UTILS "${FLANG_SOURCE_DIR}/../clang/cmake")

  # Add path for custom modules
  list(INSERT CMAKE_MODULE_PATH 0
    "${FLANG_SOURCE_DIR}/cmake"
    "${FLANG_SOURCE_DIR}/cmake/modules"
    "${LLVM_COMMON_CMAKE_UTILS}"
    "${LLVM_COMMON_CMAKE_UTILS}/Modules"
    "${LLVM_CMAKE_UTILS}"
    "${LLVM_CMAKE_UTILS}/modules"
    "${CLANG_CMAKE_UTILS}/modules"
    )

  include(AddClang)
  include(AddLLVM)
  include(AddFlang)
  include(HandleLLVMOptions)

  include(TestBigEndian)
  test_big_endian(IS_BIGENDIAN)
  if (IS_BIGENDIAN)
    add_compile_definitions(FLANG_BIG_ENDIAN=1)
  else ()
    add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
  endif ()
  include_directories(BEFORE
    ${FLANG_SOURCE_DIR}/include)
endif()

include(CheckCXXSymbolExists)
include(CheckCXXSourceCompiles)
check_cxx_symbol_exists(strerror_r string.h HAVE_STRERROR_R)
# Can't use symbol exists here as the function is overloaded in C++
check_cxx_source_compiles(
  "#include <string.h>
   int main() {
     char buf[4096];
     return strerror_s(buf, 4096, 0);
   }
  "
  HAVE_DECL_STRERROR_S)

check_cxx_compiler_flag(-fno-lto FLANG_RUNTIME_HAS_FNO_LTO_FLAG)
if (FLANG_RUNTIME_HAS_FNO_LTO_FLAG)
  set(NO_LTO_FLAGS "-fno-lto")
else()
  set(NO_LTO_FLAGS "")
endif()

configure_file(config.h.cmake config.h)
# include_directories is used here instead of target_include_directories
# because add_flang_library creates multiple objects (STATIC/SHARED, OBJECT)
# with different names
include_directories(AFTER ${CMAKE_CURRENT_BINARY_DIR})

append(${NO_LTO_FLAGS} CMAKE_C_FLAGS)
append(${NO_LTO_FLAGS} CMAKE_CXX_FLAGS)

# Disable libstdc++/libc++ assertions, even in an LLVM_ENABLE_ASSERTIONS build,
# to avoid an unwanted dependency on libstdc++/libc++.so.
add_definitions(-U_GLIBCXX_ASSERTIONS)
add_definitions(-U_LIBCPP_ENABLE_ASSERTIONS)

add_subdirectory(FortranMain)

set(sources
  ISO_Fortran_binding.cpp
  allocatable.cpp
  array-constructor.cpp
  assign.cpp
  buffer.cpp
  command.cpp
  complex-powi.cpp
  complex-reduction.c
  copy.cpp
  character.cpp
  connection.cpp
  derived.cpp
  derived-api.cpp
  descriptor.cpp
  descriptor-io.cpp
  dot-product.cpp
  edit-input.cpp
  edit-output.cpp
  environment.cpp
  exceptions.cpp
  extensions.cpp
  extrema.cpp
  file.cpp
  findloc.cpp
  format.cpp
  inquiry.cpp
  internal-unit.cpp
  iostat.cpp
  io-api.cpp
  io-error.cpp
  io-stmt.cpp
  main.cpp
  matmul-transpose.cpp
  matmul.cpp
  memory.cpp
  misc-intrinsic.cpp
  namelist.cpp
  non-tbp-dio.cpp
  numeric.cpp
  ragged.cpp
  random.cpp
  reduction.cpp
  pointer.cpp
  product.cpp
  stat.cpp
  stop.cpp
  sum.cpp
  support.cpp
  temporary-stack.cpp
  terminator.cpp
  time-intrinsic.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  type-info.cpp
  unit.cpp
  unit-map.cpp
  utf.cpp
)

option(FLANG_EXPERIMENTAL_CUDA_RUNTIME
  "Compile Fortran runtime as CUDA sources (experimental)" OFF
  )

# List of files that are buildable for all devices.
set(supported_files
  ISO_Fortran_binding.cpp
  assign.cpp
  derived.cpp
  descriptor.cpp
  stat.cpp
  terminator.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  type-info.cpp
  )

if (FLANG_EXPERIMENTAL_CUDA_RUNTIME)
  if (BUILD_SHARED_LIBS)
    message(FATAL_ERROR
      "BUILD_SHARED_LIBS is not supported for CUDA build of Fortran runtime"
      )
  endif()

  enable_language(CUDA)

  # TODO: figure out how to make target property CUDA_SEPARABLE_COMPILATION
  # work, and avoid setting CMAKE_CUDA_SEPARABLE_COMPILATION.
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

  # Treat all supported sources as CUDA files.
  set_source_files_properties(${supported_files} PROPERTIES LANGUAGE CUDA)
  set(CUDA_COMPILE_OPTIONS)
  if ("${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang")
    # Allow varargs.
    set(CUDA_COMPILE_OPTIONS
      -Xclang -fcuda-allow-variadic-functions
      )
  endif()
  if ("${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA")
    set(CUDA_COMPILE_OPTIONS
      --expt-relaxed-constexpr
      )
  endif()
  set_source_files_properties(${supported_files} PROPERTIES COMPILE_OPTIONS
    "${CUDA_COMPILE_OPTIONS}"
    )
endif()

set(FLANG_EXPERIMENTAL_OMP_OFFLOAD_BUILD "off" CACHE STRING
  "Compile Fortran runtime as OpenMP target offload sources (experimental). Valid options are 'off', 'host_device', 'nohost'")

set(FLANG_OMP_DEVICE_ARCHITECTURES "all" CACHE STRING
  "List of OpenMP device architectures to be used to compile the Fortran runtime (e.g. 'gfx1103;sm_90')")

if (NOT FLANG_EXPERIMENTAL_OMP_OFFLOAD_BUILD STREQUAL "off")
  # 'host_device' build only works with Clang compiler currently.
  # The build is done with the CMAKE_C/CXX_COMPILER, i.e. it does not use
  # the in-tree built Clang. We may have a mode that would use the in-tree
  # built Clang.
  #
  # 'nohost' is supposed to produce an LLVM Bitcode library,
  # and it has to be done with a C/C++ compiler producing LLVM Bitcode
  # compatible with the LLVM toolchain version distributed with the Flang
  # compiler.
  # In general, the in-tree built Clang should be used for 'nohost' build.
  # Note that 'nohost' build does not produce the host version of Flang
  # runtime library, so there will be two separate distributable objects.
  # 'nohost' build is a TODO.

  if (NOT FLANG_EXPERIMENTAL_OMP_OFFLOAD_BUILD STREQUAL "host_device")
    message(FATAL_ERROR "Unsupported OpenMP offload build of Flang runtime")
  endif()
  if (BUILD_SHARED_LIBS)
    message(FATAL_ERROR
      "BUILD_SHARED_LIBS is not supported for OpenMP offload build of Fortran runtime"
      )
  endif()

  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND
      "${CMAKE_C_COMPILER_ID}" MATCHES "Clang")

    set(all_amdgpu_architectures
      "gfx700;gfx701;gfx801;gfx803;gfx900;gfx902;gfx906"
      "gfx908;gfx90a;gfx90c;gfx940;gfx1010;gfx1030"
      "gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036"
      "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151"
      )
    set(all_nvptx_architectures
      "sm_35;sm_37;sm_50;sm_52;sm_53;sm_60;sm_61;sm_62"
      "sm_70;sm_72;sm_75;sm_80;sm_86;sm_89;sm_90"
      )
    set(all_gpu_architectures
      "${all_amdgpu_architectures};${all_nvptx_architectures}"
      )
    # TODO: support auto detection on the build system.
    if (FLANG_OMP_DEVICE_ARCHITECTURES STREQUAL "all")
      set(FLANG_OMP_DEVICE_ARCHITECTURES ${all_gpu_architectures})
    endif()
    list(REMOVE_DUPLICATES FLANG_OMP_DEVICE_ARCHITECTURES)

    string(REPLACE ";" "," compile_for_architectures
      "${FLANG_OMP_DEVICE_ARCHITECTURES}"
      )

    set(OMP_COMPILE_OPTIONS
      -fopenmp
      -fvisibility=hidden
      -fopenmp-cuda-mode
      --offload-arch=${compile_for_architectures}
      # Force LTO for the device part.
      -foffload-lto
      )
    set_source_files_properties(${supported_files} PROPERTIES COMPILE_OPTIONS
      "${OMP_COMPILE_OPTIONS}"
      )

    # Enable "declare target" in the source code.
    set_source_files_properties(${supported_files}
      PROPERTIES COMPILE_DEFINITIONS OMP_OFFLOAD_BUILD
      )
  else()
    message(FATAL_ERROR
      "Flang runtime build is not supported for these compilers:\n"
      "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}\n"
      "CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")
  endif()
endif()

if (NOT DEFINED MSVC)
  add_flang_library(FortranRuntime
    ${sources}
    LINK_LIBS
    FortranDecimal

    INSTALL_WITH_TOOLCHAIN
  )
else()
  add_flang_library(FortranRuntime
    ${sources}
    LINK_LIBS
    FortranDecimal
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
  add_flang_library(FortranRuntime.static ${sources}
    LINK_LIBS
    FortranDecimal.static
    INSTALL_WITH_TOOLCHAIN)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  add_flang_library(FortranRuntime.dynamic ${sources}
    LINK_LIBS
    FortranDecimal.dynamic
    INSTALL_WITH_TOOLCHAIN)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebug)
  add_flang_library(FortranRuntime.static_dbg ${sources}
    LINK_LIBS
    FortranDecimal.static_dbg
    INSTALL_WITH_TOOLCHAIN)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
  add_flang_library(FortranRuntime.dynamic_dbg ${sources}
    LINK_LIBS
    FortranDecimal.dynamic_dbg
    INSTALL_WITH_TOOLCHAIN)
  add_dependencies(FortranRuntime FortranRuntime.static FortranRuntime.dynamic
    FortranRuntime.static_dbg FortranRuntime.dynamic_dbg)
endif()
