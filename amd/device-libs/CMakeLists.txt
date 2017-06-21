##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.6)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
project(ROCm-Device-Libs)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE INTERNAL "Prefix prepended to install directories")
endif()

find_package(LLVM REQUIRED CONFIG PATHS ${LLVM_DIR} NO_DEFAULT_PATH)
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

 set(ROCM_DEVICELIB_STANDALONE_BUILD ON)
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

if (NOT DEFINED AMDGPU_TARGET_TRIPLE)
  set(AMDGPU_TARGET_TRIPLE "amdgcn--amdhsa")
endif (NOT DEFINED AMDGPU_TARGET_TRIPLE)

# Introduce GENERIC_IS_ZERO CMake option
option(GENERIC_IS_ZERO "Teach LLVM/Clang that generic address space IS address space 0 for AMDGPU target" OFF)
if (GENERIC_IS_ZERO)
  # set the target triple to amdgizcl to show to the compiler
  # that we are using a newer address space mapping
  set(AMDGPU_TARGET_TRIPLE "amdgcn--amdhsa-amdgizcl")
  # HCC will execute utils/change-addr-space.sh
  # and apply utils/add_amdgiz.sed on all .ll files in subdirectory hc/, irif/, opencl/

endif (GENERIC_IS_ZERO)

option(BUILD_HC_LIB "Build Heterogeneous Compute built-in library (hc)" ON)
option(ROCM_DEVICELIB_INCLUDE_TESTS "Build tests" ON)

if (NOT PREPARE_BUILTINS)
  add_subdirectory(utils/prepare-builtins)
  set (PREPARE_BUILTINS $<TARGET_FILE:prepare-builtins>)
endif()

include(OCL.cmake)

set(AMDGCN_LIB_LIST)
add_subdirectory(irif)
add_subdirectory(oclc)
add_subdirectory(ocml)
add_subdirectory(ockl)
add_subdirectory(opencl)

if(BUILD_HC_LIB)
  add_subdirectory(hc)
endif(BUILD_HC_LIB)

if(AMDGCN_TARGETS_LIB_LIST)
  set(${AMDGCN_TARGETS_LIB_LIST} ${AMDGCN_LIB_LIST} PARENT_SCOPE)
endif(AMDGCN_TARGETS_LIB_LIST)

if(ROCM_DEVICELIB_INCLUDE_TESTS)
  if (NOT AMDHSACOD)
    message(WARNING "amdhsacod is not found, testing disabled")
    message(WARNING "Specify amdhsacod executable with -DAMDHSACOD=<...>")
  endif()
  enable_testing()
  add_subdirectory(test/opencl)
  add_subdirectory(test/opencl_conformance)
endif(ROCM_DEVICELIB_INCLUDE_TESTS)

## CPack standard variables
set ( CPACK_PACKAGE_NAME "rocm-device-libs" )
set ( CPACK_PACKAGE_VERSION_MAJOR "0" )
set ( CPACK_PACKAGE_VERSION_MINOR "0" )
set ( CPACK_PACKAGE_VERSION_PATCH "1" )
set ( CPACK_PACKAGE_CONTACT "" )
set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute - device libraries" )
set ( CPACK_PACKAGE_DESCRIPTION "This package includes LLVM bitcode libraries." )
set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCm-Device-Libs" )

set ( CPACK_GENERATOR DEB RPM )

## Debian package values
set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "AMD" )
set ( CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" )
set ( CPACK_DEBIAN_PACKAGE_DEPENDS "" )

include ( CPack )
