##===--------------------------------------------------------------------------
##                   ROCm Device Libraries
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##===--------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13.4)

project(ROCm-Device-Libs VERSION "1.0.0")
cmake_policy(SET CMP0011 NEW)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(CMakePackageConfigHelpers)
  include(GNUInstallDirs)

  find_package(ROCM)
  if (ROCM_FOUND)
    include(ROCMSetupVersion)
    rocm_setup_version(VERSION "${PROJECT_VERSION}")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_custom_target(rocm-device-libs)

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
  find_package(LLVM REQUIRED)
  find_package(Clang HINTS ${LLVM_DIR}/../clang)

  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})

  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist CACHE INTERNAL "Prefix prepended to install directories")
  endif()

  set(ROCM_DEVICELIB_STANDALONE_BUILD ON)
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)


if (NOT DEFINED AMDGPU_TARGET_TRIPLE)
  set(AMDGPU_TARGET_TRIPLE "amdgcn-amd-amdhsa")
endif()

if (NOT PREPARE_BUILTINS)
  add_subdirectory(utils/prepare-builtins)
  set (PREPARE_BUILTINS $<TARGET_FILE:prepare-builtins>)
endif()

# Following variables are required for ROCM backwards compatibility,
# and should be removed in ROCM 7.0 release.
set(ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW "" CACHE STRING "New bitcode install location relative to CMAKE_INSTALL_PREFIX")
set(ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_OLD "" CACHE STRING "Old bitcode install location relative to CMAKE_INSTALL_PREFIX")

include(OCL)

if (NOT ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW STREQUAL "" AND
    NOT ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_OLD STREQUAL "")
  set(ROCM_DEVICE_LIBS_WRAPPER_DIR ${CMAKE_CURRENT_BINARY_DIR}/wrapper_dir)
  file(MAKE_DIRECTORY ${ROCM_DEVICE_LIBS_WRAPPER_DIR})
  add_custom_target(
      FILE_REORG_ROCM_6_0 ALL
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_NEW}
          ${ROCM_DEVICE_LIBS_WRAPPER_DIR}/${ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_OLD})
  install(
      FILES ${ROCM_DEVICE_LIBS_WRAPPER_DIR}/${ROCM_DEVICE_LIBS_BITCODE_INSTALL_LOC_OLD}
      DESTINATION .)
endif()

set(AMDGCN_LIB_LIST)
set(AMDGCN_DEP_LIST)
add_subdirectory(oclc)
add_subdirectory(ocml)
add_subdirectory(ockl)
add_subdirectory(opencl)
add_subdirectory(hip)
add_subdirectory(asanrtl)

enable_testing()
add_subdirectory(test/compile)

include(Packages)

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
  set ( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.TXT" )
  set ( CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCm-Device-Libs" )

  # Install License file
  install ( FILES "${CPACK_RESOURCE_FILE_LICENSE}"
          DESTINATION ${CMAKE_INSTALL_DOCDIR})

  set ( CPACK_GENERATOR "DEB;RPM" CACHE STRING "Default packaging generators." )

  ## ROCM version updates as per naming convention
  set ( ROCM_VERSION_FOR_PACKAGE "99999" )
  if( DEFINED ENV{ROCM_LIBPATCH_VERSION} )
    set ( ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION} )
  endif()
  ## Debian package values
  set ( CPACK_DEBIAN_PACKAGE_MAINTAINER "ROCm Compiler Support <rocm.compiler.support@amd.com>" )

  set ( CPACK_DEBIAN_PACKAGE_RELEASE "local" )
  if( DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
    set ( CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE} )
  endif()
  ## RPM package variables
  set ( CPACK_RPM_PACKAGE_RELEASE "local" )
  if( DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE} )
    set ( CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE} )
  endif()
  ## get distro for RPM package using dist
  message("device-libs CPACK_RPM_PACKAGE_RELEASE now is ${CPACK_RPM_PACKAGE_RELEASE}")
  set( CPACK_RPM_PACKAGE_LICENSE "NCSA" )

  execute_process( COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE _result_var
                 OUTPUT_VARIABLE _output_var
                 OUTPUT_STRIP_TRAILING_WHITESPACE )
  if( _result_var EQUAL "0" AND NOT _output_var STREQUAL "" )
    string (APPEND CPACK_RPM_PACKAGE_RELEASE ${_output_var})
  endif()
  # set package name as per standard
  set ( CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}" )

  set ( CPACK_RPM_FILE_NAME "RPM-DEFAULT" )
  set ( CPACK_RPM_PACKAGE_REQUIRES "rocm-core" )
  set ( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" )
  set ( CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core" )
  # Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
  if(NOT ROCM_DEP_ROCMCORE)
      string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
      string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
  endif()
  include( CPack )
endif()
