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

  # The out of tree builds of the compiler and the Fortran runtime
  # must use the same setting of FLANG_RUNTIME_F128_MATH_LIB
  # to be composable. Failure to synchronize this setting may result
  # in linking errors or fatal failures in F128 runtime functions.
  set(FLANG_RUNTIME_F128_MATH_LIB "" CACHE STRING
    "Specifies the target library used for implementing IEEE-754 128-bit float \
    math in F18 runtime, e.g. it might be libquadmath for targets where \
    REAL(16) is mapped to __float128, or libm for targets where REAL(16) \
    is mapped to long double, etc."
    )
endif()

set(linked_libraries FortranDecimal)

# function checks
find_package(Backtrace)
set(HAVE_BACKTRACE ${Backtrace_FOUND})
set(BACKTRACE_HEADER ${Backtrace_HEADER})
if(HAVE_BACKTRACE)
  list(APPEND linked_libraries ${Backtrace_LIBRARY})
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

# Check if 128-bit float computations can be done via long double.
check_cxx_source_compiles(
  "#include <cfloat>
   #if LDBL_MANT_DIG != 113
   #error LDBL_MANT_DIG != 113
   #endif
   int main() { return 0; }
  "
  HAVE_LDBL_MANT_DIG_113)

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

add_subdirectory(Float128Math)

set(sources
  ISO_Fortran_binding.cpp
  allocator-registry.cpp
  allocatable.cpp
  array-constructor.cpp
  assign.cpp
  buffer.cpp
  character.cpp
  command.cpp
  complex-powi.cpp
  complex-reduction.c
  connection.cpp
  copy.cpp
  derived-api.cpp
  derived.cpp
  descriptor-io.cpp
  descriptor.cpp
  dot-product.cpp
  edit-input.cpp
  edit-output.cpp
  environment.cpp
  exceptions.cpp
  execute.cpp
  extensions.cpp
  external-unit.cpp
  extrema.cpp
  file.cpp
  findloc.cpp
  format.cpp
  inquiry.cpp
  internal-unit.cpp
  io-api.cpp
  io-api-minimal.cpp
  io-error.cpp
  io-stmt.cpp
  iostat.cpp
  main.cpp
  matmul-transpose.cpp
  matmul.cpp
  memory.cpp
  misc-intrinsic.cpp
  namelist.cpp
  non-tbp-dio.cpp
  numeric.cpp
  pointer.cpp
  product.cpp
  pseudo-unit.cpp
  ragged.cpp
  random.cpp
  reduce.cpp
  reduction.cpp
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
  unit-map.cpp
  unit.cpp
  utf.cpp
  ${FORTRAN_MODULE_OBJECTS}
)

include(AddFlangOffloadRuntime)

# List of files that are buildable for all devices.
set(supported_files
  ISO_Fortran_binding.cpp
  allocatable.cpp
  allocator-registry.cpp
  array-constructor.cpp
  assign.cpp
  buffer.cpp
  character.cpp
  connection.cpp
  copy.cpp
  derived-api.cpp
  derived.cpp
  descriptor.cpp
  descriptor-io.cpp
  dot-product.cpp
  edit-input.cpp
  edit-output.cpp
  environment.cpp
  extrema.cpp
  external-unit.cpp
  file.cpp
  findloc.cpp
  format.cpp
  inquiry.cpp
  internal-unit.cpp
  io-api.cpp
  io-api-minimal.cpp
  io-error.cpp
  io-stmt.cpp
  iostat.cpp
  matmul-transpose.cpp
  matmul.cpp
  memory.cpp
  misc-intrinsic.cpp
  namelist.cpp
  non-tbp-dio.cpp
  numeric.cpp
  pointer.cpp
  product.cpp
  pseudo-unit.cpp
  ragged.cpp
  stat.cpp
  sum.cpp
  support.cpp
  terminator.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  type-info.cpp
  unit.cpp
  utf.cpp
  )

enable_cuda_compilation(FortranRuntime "${supported_files}")
enable_omp_offload_compilation("${supported_files}")

if (NOT TARGET FortranFloat128Math)
  # If FortranFloat128Math is not defined, then we are not building
  # standalone FortranFloat128Math library. Instead, include
  # the relevant sources into FortranRuntime itself.
  # The information is provided via FortranFloat128MathILib
  # interface library.
  get_target_property(f128_sources
    FortranFloat128MathILib INTERFACE_SOURCES
    )
  if (f128_sources)
    # The interface may define special macros for Float128Math files,
    # so we need to propagate them.
    get_target_property(f128_defs
      FortranFloat128MathILib INTERFACE_COMPILE_DEFINITIONS
      )
    set_property(SOURCE ${f128_sources}
      APPEND PROPERTY COMPILE_DEFINITIONS
      ${f128_defs}
      )
    get_target_property(f128_include_dirs
      FortranFloat128MathILib INTERFACE_INCLUDE_DIRECTORIES
      )
    set_property(SOURCE ${f128_sources}
      APPEND PROPERTY INCLUDE_DIRECTORIES
      ${f128_include_dirs}
      )
    list(APPEND sources ${f128_sources})
  endif()
endif()

if (NOT DEFINED MSVC)
  add_flang_library(FortranRuntime
    ${sources}
    LINK_LIBS
    ${linked_libraries}

    INSTALL_WITH_TOOLCHAIN
  )
else()
  add_flang_library(FortranRuntime
    ${sources}
    LINK_LIBS
    ${linked_libraries}
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
  add_flang_library(FortranRuntime.static ${sources}
    LINK_LIBS
    FortranDecimal.static
    INSTALL_WITH_TOOLCHAIN)
  set_target_properties(FortranRuntime.static PROPERTIES FOLDER "Flang/Runtime Libraries")
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  add_flang_library(FortranRuntime.dynamic ${sources}
    LINK_LIBS
    FortranDecimal.dynamic
    INSTALL_WITH_TOOLCHAIN)
  set_target_properties(FortranRuntime.dynamic PROPERTIES FOLDER "Flang/Runtime Libraries")
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebug)
  add_flang_library(FortranRuntime.static_dbg ${sources}
    LINK_LIBS
    FortranDecimal.static_dbg
    INSTALL_WITH_TOOLCHAIN)
  set_target_properties(FortranRuntime.static_dbg PROPERTIES FOLDER "Flang/Runtime Libraries")
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
  add_flang_library(FortranRuntime.dynamic_dbg ${sources}
    LINK_LIBS
    FortranDecimal.dynamic_dbg
    INSTALL_WITH_TOOLCHAIN)
  set_target_properties(FortranRuntime.dynamic_dbg PROPERTIES FOLDER "Flang/Runtime Libraries")
  add_dependencies(FortranRuntime FortranRuntime.static FortranRuntime.dynamic
    FortranRuntime.static_dbg FortranRuntime.dynamic_dbg)
endif()
set_target_properties(FortranRuntime PROPERTIES FOLDER "Flang/Runtime Libraries")

# If FortranRuntime is part of a Flang build (and not a separate build) then
# add dependency to make sure that Fortran runtime library is being built after
# we have the Flang compiler available.  This also includes the MODULE files
# that compile when the 'flang' target is built.
#
# TODO: This is a workaround and should be updated when runtime build procedure
# is changed to a regular runtime build.  See discussion in PR #95388.
if (TARGET flang AND TARGET module_files)
  add_dependencies(FortranRuntime flang module_files)
endif()

if (FLANG_CUF_RUNTIME)
  add_subdirectory(CUDA)
endif()
