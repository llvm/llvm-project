#===-- runtime/CMakeLists.txt ----------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.13.4)
  if ("${CMAKE_VERSION}" VERSION_LESS "3.20.0")
    message(WARNING
      "Your CMake version is ${CMAKE_VERSION}. Starting with LLVM 17.0.0, the "
      "minimum version of CMake required to build LLVM will become 3.20.0, and "
      "using an older CMake will become an error. Please upgrade your CMake to "
      "at least 3.20.0 now to avoid issues in the future!")
  endif()

  project(FlangRuntime C CXX)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_EXTENSIONS OFF)

  set(FLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

  set(LLVM_COMMON_CMAKE_UTILS "${FLANG_SOURCE_DIR}/../cmake")
  set(LLVM_CMAKE_UTILS "${FLANG_SOURCE_DIR}/../llvm/cmake")

  # Add path for custom modules
  list(INSERT CMAKE_MODULE_PATH 0
    "${FLANG_SOURCE_DIR}/cmake"
    "${FLANG_SOURCE_DIR}/cmake/modules"
    "${LLVM_COMMON_CMAKE_UTILS}"
    "${LLVM_COMMON_CMAKE_UTILS}/Modules"
    "${LLVM_CMAKE_UTILS}"
    "${LLVM_CMAKE_UTILS}/modules"
    )

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
check_cxx_symbol_exists(strerror string.h HAVE_STRERROR)
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

if (NOT (HAVE_STRERROR OR HAVE_STRERROR_R OR HAVE_DECL_STRERROR_S))
  message(FATAL_ERROR "None of strerror, strerror_r, strerror_s found.")
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

add_flang_library(FortranRuntime
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
  terminator.cpp
  time-intrinsic.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  type-info.cpp
  unit.cpp
  unit-map.cpp
  utf.cpp

  LINK_LIBS
  FortranDecimal

  INSTALL_WITH_TOOLCHAIN
)
