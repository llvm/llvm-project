#===-- runtime/CMakeLists.txt ----------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

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

if (NOT (HAVE_STRERROR OR HAVE_STRERROR_R OR HAVE_DECL_STRERROR_S))
  message(FATAL_ERROR "None of strerror, strerror_r, strerror_s found.")
endif()

configure_file(config.h.cmake config.h)
# include_directories is used here instead of target_include_directories
# because add_flang_library creates multiple objects (STATIC/SHARED, OBJECT)
# with different names
include_directories(AFTER ${CMAKE_CURRENT_BINARY_DIR})

add_flang_library(FortranRuntime
  ISO_Fortran_binding.cpp
  allocatable.cpp
  buffer.cpp
  character.cpp
  connection.cpp
  derived-type.cpp
  descriptor.cpp
  edit-input.cpp
  edit-output.cpp
  environment.cpp
  file.cpp
  format.cpp
  internal-unit.cpp
  iostat.cpp
  io-api.cpp
  io-error.cpp
  io-stmt.cpp
  main.cpp
  memory.cpp
  stop.cpp
  terminator.cpp
  tools.cpp
  transformational.cpp
  type-code.cpp
  unit.cpp
  unit-map.cpp

  LINK_LIBS
  FortranDecimal
)
