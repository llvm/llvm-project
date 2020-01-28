#===-- CMakeLists.txt ------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

cmake_minimum_required(VERSION 3.9.0)

option(LINK_WITH_FIR "Link driver with FIR and LLVM" ON)

# Pass -DGCC=... to cmake to use a specific gcc installation.
if( GCC )
  set(CMAKE_CXX_COMPILER "${GCC}/bin/g++")
  set(CMAKE_CC_COMPILER "${GCC}/bin/gcc")
  set(CMAKE_BUILD_RPATH "${GCC}/lib64")
  set(CMAKE_INSTALL_RPATH "${GCC}/lib64")
endif()
if(BUILD_WITH_CLANG)
  file(TO_CMAKE_PATH "${BUILD_WITH_CLANG}" CLANG_PATH)
  set(CMAKE_CXX_COMPILER "${CLANG_PATH}/bin/clang++")
  set(CMAKE_CC_COMPILER "${CLANG_PATH}/bin/clang")
  if(GCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC}")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstring-conversion")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcovered-switch-default")
endif()

# Set RPATH in every executable, overriding the default setting.
# If you set this first variable back to true (the default),
# also set the second one.
set(CMAKE_SKIP_BUILD_RPATH false)
set(CMAKE_BUILD_WITH_INSTALL_RPATH false)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" "${CMAKE_INSTALL_RPATH}")

# Reminder: Setting CMAKE_CXX_COMPILER must be done before calling project()

project(f18 CXX)

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug )
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}" )

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})

include(AddLLVM)

# Get names for the LLVM libraries
#
# The full list of LLVM components can be obtained with
#
#   llvm-config --components
#
# Similarly, the (static) libraries corresponding to some
# components (default is 'all') can be obtained with
#
#   llvm-config --libs --link-static [component ...]
#
# See also
#  http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
#  https://stackoverflow.com/questions/41924375/llvm-how-to-specify-all-link-libraries-as-input-to-llvm-map-components-to-libna
#  https://stackoverflow.com/questions/33948633/how-do-i-link-when-building-with-llvm-libraries

# Add LLVM include files as if they were SYSTEM because there are complex unused
# parameter issues that may or may not appear depending on the environments and
# compilers (ifdefs are involved). This allows warnings from LLVM headers to be
# ignored while keeping -Wunused-parameter a fatal error inside f18 code base.
# This may have to be fine-tuned if flang headers are consider part of this
# LLVM_INCLUDE_DIRS when merging in the monorepo (Warning from flang headers
# should not be suppressed).
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# LLVM_LIT_EXTERNAL store in cache so it could be used by AddLLVM.cmake
set(LLVM_EXTERNAL_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit CACHE STRING "Command used to spawn lit")

if(LINK_WITH_FIR)
  message(STATUS "Linking driver with FIR and LLVM")
  llvm_map_components_to_libnames(LLVM_COMMON_LIBS support)
  message(STATUS "LLVM libraries: ${LLVM_COMMON_LIBS}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
   if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      if(BUILD_WITH_CLANG_LIBRARIES)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++ -I${BUILD_WITH_CLANG_LIBRARIES}/include/c++/v1 -DCLANG_LIBRARIES")
         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -Wl,-rpath,${BUILD_WITH_CLANG_LIBRARIES}/lib -L${BUILD_WITH_CLANG_LIBRARIES}/lib")
      else()
         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")
      endif()
      if(GCC)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC}")
      endif()
   endif()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Werror")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wimplicit-fallthrough")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdelete-non-virtual-dtor")
   set(CMAKE_CXX_FLAGS_RELEASE    "-O2")
   set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2 '-DCHECK=(void)'")
   set(CMAKE_CXX_FLAGS_DEBUG      "-g -DDEBUGF18")

   # Building shared libraries is death on performance with GCC by default
   # due to the need to preserve the right to override external entry points
   # at dynamic link time.  -fno-semantic-interposition waives that right and
   # recovers a little bit of that performance.
   if (BUILD_SHARED_LIBS AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-semantic-interposition")
   endif()
endif()

set(FLANG_VERSION_MAJOR      "0")
set(FLANG_VERSION_MINOR      "1")
set(FLANG_VERSION_PATCHLEVEL "0")
set(FLANG_VERSION "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}.${FLANG_VERSION_PATCHLEVEL}")
message(STATUS "FLANG version: ${FLANG_VERSION}")

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FLANG_TOOLS_DIR  ${FLANG_BINARY_DIR}/tools/f18/bin)

include_directories(BEFORE
  ${FLANG_BINARY_DIR}/include
  ${FLANG_SOURCE_DIR}/include
  )

enable_testing()

add_subdirectory(include/flang)
add_subdirectory(lib)
add_subdirectory(runtime)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(test-lit)

configure_file(
  ${FLANG_SOURCE_DIR}/include/flang/Config/config.h.cmake
  ${FLANG_BINARY_DIR}/include/flang/Config/config.h)
