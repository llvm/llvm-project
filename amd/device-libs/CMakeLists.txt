##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.6)

project(ROCm-Device-Libs)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE INTERNAL "Prefix prepended to install directories")
endif()

if (AMDHSACOD)
else()
  message(WARNING "amdhsacod is not found, testing disabled")
  message(WARNING "Specify amdhsacod executable with -DAMDHSACOD=<...>")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

find_package(LLVM REQUIRED PATHS ${LLVM_DIR} "/opt/rocm/llvm" NO_DEFAULT_PATH)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

#include_directories(${CMAKE_SOURCE_DIR}/inc)

list(APPEND CMAKE_OCL_SOURCE_FILE_EXTENSIONS .cl)
list(APPEND CMAKE_C_SOURCE_FILE_EXTENSIONS .cl)
list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS .cl)

set (CLANG_OCL_FLAGS "-Werror -x cl -Xclang -cl-std=CL2.0 -fblocks -target amdgcn--amdhsa -DCL_VERSION_2_0=200 -D__OPENCL_C_VERSION__=200 -Dcl_khr_fp64 -Dcl_khr_fp16 -Dcl_khr_subgroups -Xclang -finclude-default-header ${CLANG_OPTIONS_APPEND}")
set (CLANG_OCL_LINK_FLAGS "-target amdgcn--amdhsa -mcpu=fiji")

include(OCL.cmake)

add_subdirectory(utils/prepare-builtins)
add_subdirectory(irif)
add_subdirectory(oclc)
add_subdirectory(ocml)
add_subdirectory(ockl)
add_subdirectory(opencl)
add_subdirectory(hc)
add_subdirectory(test/opencl)
add_subdirectory(test/opencl_conformance)

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
