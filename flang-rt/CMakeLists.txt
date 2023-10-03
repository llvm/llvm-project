# CMake build for the Flang runtime libraries
# The source for the flang runtime libraries (FortranDecimalRT, FortranRuntime)
# exist in the flang top-level directory.
# Flang-rt is only scaffolding and does not provide any additional source files.

cmake_minimum_required(VERSION 3.20.0)

#===============================================================================
# Configure CMake
#===============================================================================
set(LLVM_COMMON_CMAKE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(${LLVM_COMMON_CMAKE_UTILS}/Modules/CMakePolicy.cmake
  NO_POLICY_SCOPE)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/flang-rt/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/flang-rt/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${CMAKE_BINARY_DIR}/flang-rt/lib)
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_BINARY_DIR}/flang-rt)

set(FLANG_RT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FLANG_RT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

#===============================================================================
# Setup Options and Defaults
#===============================================================================
option(FLANG_RT_ENABLE_SHARED "Build flang-rt as a shared library." OFF)
option(FLANG_RT_ENABLE_STATIC "Build flang-rt as a static library." OFF)

option(FLANG_RT_INCLUDE_TESTS
    "Generate build targets for the Flang-rt unit tests." ${LLVM_INCLUDE_TESTS})

# MLIR_DIR must be passed on invocation of flang-rt because it is needed for the Flang package.
if(NOT DEFINED MLIR_DIR OR MLIR_DIR STREQUAL "")
  message(FATAL_ERROR "MLIR_DIR must be set to the directory of the MLIRConfig cmake file in order to find the MLIR package.")
endif()
# Flang-rt requires a pre-built/installed version of flang that requires MLIR.
find_package(MLIR REQUIRED HINTS "${MLIR_DIR}")

# FLANG_DIR must be passed on invocation of flang-rt.
if(NOT DEFINED FLANG_DIR OR FLANG_DIR STREQUAL "")
  message(FATAL_ERROR "FLANG_DIR must be set to the directory of the FlangConfig cmake file in order to find the Flang package.")
endif()
# Flang-rt requires a pre-built/installed version of flang.
# Flang-rt uses flang/Common headers.
# Finding this package exposes FLANG_SOURCE_DIR, FLANG_BINARY_DIR, and FLANG_INCLUDE_DIRS to Flang-rt
find_package(Flang REQUIRED HINTS "${FLANG_DIR}")
# If the user specifies a relative path to LLVM_DIR, the calls to include
# LLVM modules fail. Append the absolute path to LLVM_DIR instead.
get_filename_component(FLANG_DIR_ABSOLUTE ${FLANG_DIR} REALPATH)
list(APPEND CMAKE_MODULE_PATH ${FLANG_DIR_ABSOLUTE})

set(LLVM_COMMON_CMAKE_UTILS "${FLANG_RT_SOURCE_DIR}/../cmake")
set(LLVM_CMAKE_UTILS "${LLVM_BUILD_MAIN_SOURCE_DIR}/cmake")
set(CLANG_CMAKE_UTILS "${FLANG_RT_SOURCE_DIR}/../clang/cmake")

if (FLANG_RT_INCLUDE_TESTS)
  # LLVM_DIR must be passed on invocation of flang-rt when tests are enabled.
  if(NOT DEFINED LLVM_DIR OR LLVM_DIR STREQUAL "")
    message(FATAL_ERROR "LLVM_DIR must be set to the directory of the LLVMConfig cmake file in order to find the LLVM package.")
  endif()
  # We need a pre-built/installed version of LLVM for gtest.
  find_package(LLVM REQUIRED HINTS "${LLVM_DIR}")
  # If the user specifies a relative path to LLVM_DIR, the calls to include
  # LLVM modules fail. Append the absolute path to LLVM_DIR instead.
  get_filename_component(LLVM_DIR_ABSOLUTE ${LLVM_DIR} REALPATH)
  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR_ABSOLUTE})

  add_compile_definitions(FLANG_RT_INCLUDE_TESTS=1)

  if (DEFINED FLANG_BINARY_DIR)
    set(FLANG_BINARY_DIR ${FLANG_BINARY_DIR} CACHE PATH "Path to the Flang build directory" FORCE)
  else()
    message(FATAL_ERROR "FLANG_BINARY_DIR must be defined or passed by the user when building tests.")
  endif()
  set(FLANG_RT_TEST_COMPILER ${FLANG_BINARY_DIR}/bin/flang-new
      CACHE PATH "Compiler to use for testing")
  set(FLANG_RT_GTEST_AVAIL 1)
endif()

# Add path for custom modules
list(INSERT CMAKE_MODULE_PATH 0
  "${FLANG_SOURCE_DIR}/cmake"
  "${FLANG_SOURCE_DIR}/cmake/modules"
  "${CLANG_CMAKE_UTILS}/modules"
  "${LLVM_CMAKE_UTILS}/modules"
)

include(AddClang)
include(AddFlang)
include(TestBigEndian)
test_big_endian(IS_BIGENDIAN)
if (IS_BIGENDIAN)
  add_compile_definitions(FLANG_BIG_ENDIAN=1)
else ()
  add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
endif ()

# Flang's include directories are needed for flang/Common.
include_directories(SYSTEM ${FLANG_INCLUDE_DIRS})

# LLVM's include directories are needed for gtest.
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

#===============================================================================
# Add Subdirectories
#===============================================================================
set(FORTRAN_DECIMAL_SRC "${FLANG_SOURCE_DIR}/lib/Decimal")
set(FORTRAN_RUNTIME_SRC "${FLANG_SOURCE_DIR}/runtime")
set(FORTRAN_MAIN_SRC "${FLANG_SOURCE_DIR}/runtime/FortranMain")

add_subdirectory(${FORTRAN_DECIMAL_SRC} FortranDecimalRT)
add_subdirectory(${FORTRAN_RUNTIME_SRC} FortranRuntime)
add_subdirectory(${FORTRAN_MAIN_SRC} FortranMain)

if (FLANG_RT_INCLUDE_TESTS)
  add_subdirectory(test)
  if (FLANG_RT_GTEST_AVAIL)
    add_subdirectory(unittests)
  endif()
endif()

#===============================================================================
# Create Flang-rt wrapper library
#===============================================================================
# Build as shared by default if no linkage type option set.
if (NOT FLANG_RT_ENABLE_SHARED AND NOT FLANG_RT_ENABLE_STATIC)
  add_library(flang-rt SHARED $<TARGET_OBJECTS:obj.FortranDecimalRT>
                              $<TARGET_OBJECTS:obj.FortranRuntime>)
endif()
if (FLANG_RT_ENABLE_SHARED)
  add_library(flang-rt SHARED $<TARGET_OBJECTS:obj.FortranDecimalRT>
                              $<TARGET_OBJECTS:obj.FortranRuntime>)
endif()
if (FLANG_RT_ENABLE_STATIC AND NOT FLANG_RT_ENABLE_SHARED)
  add_library(flang-rt STATIC $<TARGET_OBJECTS:obj.FortranDecimalRT>
                              $<TARGET_OBJECTS:obj.FortranRuntime>)
endif()
# When building both static and shared, we need to append _static to the name
# to avoid naming conflicts.
if (FLANG_RT_ENABLE_STATIC AND FLANG_RT_ENABLE_SHARED)
  add_library(flang-rt_static STATIC $<TARGET_OBJECTS:obj.FortranDecimalRT>
                                     $<TARGET_OBJECTS:obj.FortranRuntime>)
endif()
