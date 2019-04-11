##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)

cmake_policy(SET CMP0011 NEW)

# Build ROCM-Device-Libs with ccache if the package is present.
set(ROCM_DEVICE_LIBS_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if(ROCM_DEVICE_LIBS_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(FATAL_ERROR "Unable to find the program ccache. Set ROCM_DEVICE_LIBS_CCACHE_BUILD to OFF")
  endif()
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
project(ROCm-Device-Libs)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE INTERNAL "Prefix prepended to install directories")
endif()

find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

 set(ROCM_DEVICELIB_STANDALONE_BUILD ON)
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

if (NOT DEFINED AMDGPU_TARGET_TRIPLE)
  set(AMDGPU_TARGET_TRIPLE "amdgcn-amd-amdhsa")
endif()

option(BUILD_HC_LIB "Build Heterogeneous Compute built-in library (hc)" ON)
option(ROCM_DEVICELIB_INCLUDE_TESTS "Build tests" ON)

if (NOT PREPARE_BUILTINS)
  add_subdirectory(utils/prepare-builtins)
  set (PREPARE_BUILTINS $<TARGET_FILE:prepare-builtins>)
endif()

include(OCL.cmake)

set(AMDGCN_LIB_LIST)
set(AMDGCN_DEP_LIST)
add_subdirectory(irif)
add_subdirectory(oclc)
add_subdirectory(ocml)
add_subdirectory(ockl)
add_subdirectory(opencl)
add_subdirectory(hip)

if(BUILD_HC_LIB)
  add_subdirectory(hc)
endif()

if(AMDGCN_TARGETS_LIB_LIST)
  set(${AMDGCN_TARGETS_LIB_LIST} ${AMDGCN_LIB_LIST} PARENT_SCOPE)
endif()

if(AMDGCN_TARGETS_LIB_DEPS)
  set(${AMDGCN_TARGETS_LIB_DEPS} ${AMDGCN_DEP_LIST} PARENT_SCOPE)
endif()

include(Packages.cmake)

if(ROCM_DEVICELIB_INCLUDE_TESTS)
  enable_testing()
  add_subdirectory(test/opencl)
  add_subdirectory(test/opencl_conformance)
endif()

## CPack standard variables
set ( CPACK_PACKAGE_NAME "rocm-device-libs" )
set ( CPACK_PACKAGE_VERSION_MAJOR "0" )
set ( CPACK_PACKAGE_VERSION_MINOR "0" )
set ( CPACK_PACKAGE_VERSION_PATCH "1" )
set ( CPACK_PACKAGE_CONTACT "" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute - device libraries" )
set ( CPACK_PACKAGE_DESCRIPTION "This package includes LLVM bitcode libraries." )
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCm-Device-Libs" )

set ( CPACK_GENERATOR "DEB;RPM" CACHE STRING "Default packaging generators." )

## Debian package values
set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "AMD" )
set ( CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" )
set ( CPACK_DEBIAN_PACKAGE_DEPENDS "" )

include ( CPack )
