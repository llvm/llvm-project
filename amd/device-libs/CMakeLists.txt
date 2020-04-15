##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.4.3)

project(ROCm-Device-Libs VERSION "1.0.0")
cmake_policy(SET CMP0011 NEW)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(CMakePackageConfigHelpers)
  include(GNUInstallDirs)

  find_package(ROCM PATHS "/opt/rocm")
  if (ROCM_FOUND)
    include(ROCMSetupVersion)
    rocm_setup_version(VERSION "${PROJECT_VERSION}")
  endif()
endif()


# Optionally, build Device Libs with ccache.
set(ROCM_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if (ROCM_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if (CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(WARNING "Unable to find ccache. Falling back to real compiler")
  endif() # if (CCACHE_PROGRAM)
endif() # if (ROCM_CCACHE_BUILD)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

  find_package(LLVM REQUIRED PATHS ${LLVM_DIR} "/opt/rocm/llvm")
  find_package(Clang REQUIRED PATHS ${Clang_DIR} ${LLVM_DIR}/../clang "/opt/rocm/llvm")
  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})

  find_program(CLANG "clang" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  find_program(LLVM_LINK "llvm-link" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  find_program(LLVM_OBJDUMP "llvm-objdump" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  find_program(LLVM_OPT "opt" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE INTERNAL "Prefix prepended to install directories")
  endif()

  set(ROCM_DEVICELIB_STANDALONE_BUILD ON)
else()
  # Included as part of llvm build with -DLLVM_EXTERNAL_PROJECTS.
  set(CLANG $<TARGET_FILE:clang>)
  set(LLVM_LINK $<TARGET_FILE:llvm-link>)
  set(LLVM_OBJDUMP $<TARGET_FILE:llvm-objdump>)
  set(LLVM_OPT $<TARGET_FILE:opt>)
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)


if (NOT DEFINED AMDGPU_TARGET_TRIPLE)
  set(AMDGPU_TARGET_TRIPLE "amdgcn-amd-amdhsa")
endif()

option(BUILD_HC_LIB "Build Heterogeneous Compute built-in library (hc)" ON)

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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  ## CPack standard variables
  set ( CPACK_PACKAGE_NAME "rocm-device-libs" )
  set ( CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}" )
  set ( CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}" )
  set ( CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}" )
  set ( CPACK_PACKAGE_VERSION "${PROJECT_VERSION}" )
  set ( CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc." )
  set ( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute - device libraries" )
  set ( CPACK_PACKAGE_DESCRIPTION "This package includes LLVM bitcode libraries." )
  set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCm-Device-Libs" )

  set ( CPACK_GENERATOR "DEB;RPM" CACHE STRING "Default packaging generators." )

  ## Debian package values
  set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "AMD" )
  set ( CPACK_PACKAGING_INSTALL_PREFIX "/opt/rocm" CACHE PATH "CPACK packaging path prefix default" )
  set ( CPACK_DEBIAN_PACKAGE_DEPENDS "" )

  include( CPack )
endif()
