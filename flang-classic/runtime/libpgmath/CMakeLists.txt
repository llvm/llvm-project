#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

cmake_minimum_required(VERSION 3.9.0)

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # Set MACOSX_RPATH=YES by default
endif()
if(POLICY CMP0022)
  cmake_policy(SET CMP0022 NEW) # Required when interacting with LLVM and Clang
endif()

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH})

# Enable build of generic math functions
set(LIBPGMATH_WITH_GENERIC FALSE CACHE BOOL "Build using generic?")

set(PACKAGE_NAME libpgmath)

# Standalone build or part of LLVM?
set(LIBPGMATH_STANDALONE_BUILD FALSE)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(pgmath C CXX ASM)

  set(LIBPGMATH_STANDALONE_BUILD TRUE)
endif()

if (LIBPGMATH_STANDALONE_BUILD)
  include(FindPython3)
  if( NOT Python3_Interpreter_FOUND )
    message(WARNING "Failed to find Python 3 interpreter. "
                    "Libpgmath test suite will be disabled.")
    set(LLVM_INCLUDE_TESTS OFF)
  endif()
endif()

set(LIBPGMATH_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
if ("${LIBPGMATH_SYSTEM_NAME}" STREQUAL "OpenBSD")
  set(LIBPGMATH_SYSTEM_NAME "Linux")
endif ()

set(LIBPGMATH_SYSTEM_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if ("${LIBPGMATH_SYSTEM_PROCESSOR}" STREQUAL "AMD64" OR
    "${LIBPGMATH_SYSTEM_PROCESSOR}" STREQUAL "amd64" )
    set(LIBPGMATH_SYSTEM_PROCESSOR "x86_64")
elseif ("${LIBPGMATH_SYSTEM_PROCESSOR}" STREQUAL "ARM64")
    set(LIBPGMATH_SYSTEM_PROCESSOR "aarch64")
endif ()

option(FLANG_ENABLE_QUADFP "Enable support for quad-precision floating-point math on AArch64" ON)
set(TARGET_SUPPORTS_QUADFP False)
if(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  if ("${LLVM_FLANG_CPU_TARGET}" STREQUAL "")
    message(STATUS "Setting libpgmath CPU target to 'native'")
    set(LLVM_FLANG_CPU_TARGET "native")
  else()
    message(STATUS "Libpgmath CPU target explicitly set to ${LLVM_FLANG_CPU_TARGET}")
  endif()
  if(FLANG_ENABLE_QUADFP AND NOT ("${LIBPGMATH_SYSTEM_NAME}" STREQUAL "Windows"))
    add_definitions(-DTARGET_SUPPORTS_QUADFP)
    set(TARGET_SUPPORTS_QUADFP True)
  endif()
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND ${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64|aarch64")
  if(CMAKE_C_COMPILER_VERSION VERSION_LESS "7.1.0")
    message(FATAL_ERROR "Found gcc at " ${CMAKE_C_COMPILER} " version " ${CMAKE_C_COMPILER_VERSION} ", but version 7.1.0 or newer required.")
  else()
    message(STATUS "Found GCC Version ${CMAKE_C_COMPILER_VERSION}.")
  endif()
  # elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  #   if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "your.required.msvc.version")
  #     message(FATAL_ERROR "Insufficient msvc version")
  #   endif()
  # elseif(...)
  # # etc.
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND ${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64")
  string(REPLACE "-O2" "-O3 -finline-functions -funroll-loops" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-O2" "-O3 -finline-functions -funroll-loops" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-std=c++11" "-std=gnu++11" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fno-tree-vectorize" "-ftree-vectorize" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-fno-tree-vectorize" "-ftree-vectorize" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fno-tree-slp-vectorize" "-ftree-slp-vectorize" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-fno-tree-slp-vectorize" "-ftree-slp-vectorize" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fno-reorder-blocks" "-freorder-blocks" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-fno-reorder-blocks" "-freorder-blocks" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fno-toplevel-reorder" "-ftoplevel-reorder" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-fno-toplevel-reorder" "-ftoplevel-reorder -felide-constructors" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-std=c++11" "-std=gnu++11" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "-std=c++11" "-std=gnu++11" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

# Needs to be changed to support cross-compilation
include(GetHostTriple)
get_host_triple(LIBPGMATH_HOST_TRIPLE)

# Setting directory names
set(LIBPGMATH_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBPGMATH_SRC_DIR ${LIBPGMATH_BASE_DIR}/lib)
set(LIBPGMATH_TOOLS_DIR ${LIBPGMATH_BASE_DIR}/tools)
set(LIBPGMATH_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBPGMATH_RUNTIME_PATH ${CMAKE_BINARY_DIR}/lib)
set(LIBPGMATH_LIBRARY_NAME pgmath)
set(LIBPGMATH_RTL ${CMAKE_SHARED_LIBRARY_PREFIX}${LIBPGMATH_LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Require out of source build.
include(MacroEnsureOutOfSourceBuild)
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate
 build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )

# Support functions
include(LibmathUtils)

# Setup Source Code And Tests
add_subdirectory(lib)
add_subdirectory(test)

if(0)
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
endif()
