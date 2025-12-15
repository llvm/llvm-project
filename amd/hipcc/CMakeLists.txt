cmake_minimum_required(VERSION 3.13.4)
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()

project(hipcc VERSION "1.1.1" LANGUAGES C CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

include( utils.cmake )
set( HIPCC "hipcc" )
set( HIPCC_NVIDIA "hipcc-nvidia")
set( COMP_TYPE "AMD" )
set( COMP_TYPE_NVIDIA "NVIDIA" )
set( BUILD_ENABLE_LINTIAN_OVERRIDES ON CACHE BOOL "Enable/Disable Lintian Overrides" )
set( BUILD_DEBIAN_PKGING_FLAG ON CACHE BOOL "Internal Status Flag to indicate Debian Packaging Build" )

# Generate static package, when BUILD_SHARED_LIBS is set to OFF.
# Default to ON
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(ADDITIONAL_SHARED_LIBRARIES_TO_LINK
  libstdc++fs.so)

set(HIPCC_BIN
  hipcc)
set(HIPCC_SOURCES
  src/hipcc.cpp
  src/utils.cpp
)

set(HIPCONFIG_BIN
  hipconfig)
set(HIPCONFIG_SOURCES
  src/hipconfig.cpp
  src/utils.cpp
)

add_executable(${HIPCC_BIN} ${HIPCC_SOURCES})
if(NOT WIN32)
  # C++17 does not require std lib linking.
  target_link_libraries(${HIPCC_BIN} ${ADDITIONAL_SHARED_LIBRARIES_TO_LINK})
endif()

add_executable(${HIPCONFIG_BIN} ${HIPCONFIG_SOURCES})
if(NOT WIN32)
  # C++17 does not require std lib linking.
  target_link_libraries(${HIPCONFIG_BIN} ${ADDITIONAL_SHARED_LIBRARIES_TO_LINK})
endif()

# Copy scripts and batch files to build directory.
file(COPY ${PROJECT_SOURCE_DIR}/bin/ DESTINATION ${PROJECT_BINARY_DIR})

# Packaging:
set(PKG_MAINTAINER_NM  "ROCm Compiler Support")
set(PKG_MAINTAINER_EMAIL  "rocm.compiler.support@amd.com")
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL AMD NVIDIA)
set(CPACK_GENERATOR "DEB;RPM;ZIP" CACHE STRING "Default packaging generators")
set(CPACK_PACKAGE_CONTACT "${PKG_MAINTAINER_NM} <${PKG_MAINTAINER_EMAIL}>")
set(CPACK_PACKAGE_DESCRIPTION "HIP Compiler Driver")
set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
set(CPACK_PACKAGE_VERSION_MAJOR "${hipcc_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${hipcc_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${hipcc_VERSION_PATCH}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

# Debian specific packaging variables.
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-core")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/ROCm/llvm-project/tree/amd-staging/amd/hipcc")
if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
else()
  set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
endif()

# AMD specific Debian packaging varibles.
set(CPACK_DEBIAN_AMD_PACKAGE_NAME "hipcc")
set(CPACK_DEBIAN_AMD_PACKAGE_DEPENDS "rocm-core, rocm-llvm")

# NVIDIA specific Debian packaging variables.
set(CPACK_DEBIAN_NVIDIA_PACKAGE_NAME "hipcc-nvidia")
set(CPACK_DEBIAN_NVIDIA_PACKAGE_DEPENDS "rocm-core") # for NVIDIA we don't need to add rocm-llvm as a dependency


# RPM specific packaging variables.
set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")
set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core")
set(CPACK_RPM_PACKAGE_AUTOREQPROV 0)
if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
  set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
else()
  set(CPACK_RPM_PACKAGE_RELEASE "local")
endif()
if(CPACK_RPM_PACKAGE_RELEASE)
  set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
endif()

# AMD specific RPM packaging varibables.
set(CPACK_RPM_AMD_PACKAGE_NAME "hipcc")
set(CPACK_RPM_AMD_PACKAGE_REQUIRES "rocm-core, rocm-llvm")

# NVIDIA specific RPM packaging variables.
set(CPACK_RPM_NVIDIA_PACKAGE_NAME "hipcc-nvidia")
set(CPACK_RPM_NVIDIA_PACKAGE_REQUIRES "rocm-core") # for NVIDIA we don't need to add rocm-llvm as a dependency

# ROCM versioning.
set(ROCM_VERSION_FOR_PACKAGE "")
if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
  set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
elseif(DEFINED ENV{ROCM_VERSION})
  string(REGEX REPLACE "." "" ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_VERSION})
else()
  set(ROCM_VERSION_FOR_PACKAGE "99999")
endif()
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")

# Exclude Windows specific BAT scripts from install/packaging for Linux.
if (NOT WIN32)
  # AMD specific.
  file(GLOB install_bin_files "${CMAKE_CURRENT_SOURCE_DIR}/bin/*")
  foreach(ITEM ${install_bin_files})
    # Exclude *.bat files
    if(NOT "${ITEM}" MATCHES ".bat$")
      # For *.pm files not require binary permissions
      if("${ITEM}" MATCHES ".pm$")
        LIST( APPEND gen_files "${ITEM}" )
      else()
        LIST( APPEND bin_files "${ITEM}" )
      endif()
    endif()
  endforeach()
  # Install with right type and Permissions
  install(PROGRAMS ${bin_files}
    DESTINATION ./bin
    COMPONENT AMD)
  install(FILES ${gen_files}
    DESTINATION ./bin
    COMPONENT AMD)

  # NVIDIA specific.
  install(PROGRAMS ${bin_files}
    DESTINATION ./bin
    COMPONENT NVIDIA)
  install(FILES ${gen_files}
    DESTINATION ./bin
    COMPONENT NVIDIA)
else ()
  # AMD specific.
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    DESTINATION .
    USE_SOURCE_PERMISSIONS
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT AMD)

  # NVIDIA specific.
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    DESTINATION .
    USE_SOURCE_PERMISSIONS
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT NVIDIA)
endif()

# AMD specific.
install(FILES
  "LICENSE.txt"
  "README.md"
  COMPONENT AMD
  DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(TARGETS ${HIPCC_BIN}
        COMPONENT AMD
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS ${HIPCONFIG_BIN}
        COMPONENT AMD
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_pkg( ${HIPCC} ${COMP_TYPE} ${CPACK_PACKAGE_VERSION} ${PKG_MAINTAINER_NM} ${PKG_MAINTAINER_EMAIL} )

# NVIDIA specific.
install(FILES
  "LICENSE.txt"
  "README.md"
  COMPONENT NVIDIA
  DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(TARGETS ${HIPCC_BIN}
        COMPONENT NVIDIA
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS ${HIPCONFIG_BIN}
        COMPONENT NVIDIA
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_pkg( ${HIPCC_NVIDIA} ${COMP_TYPE_NVIDIA} ${CPACK_PACKAGE_VERSION} ${PKG_MAINTAINER_NM} ${PKG_MAINTAINER_EMAIL} )

# TODO: WIN32 check need to be removed if backward
# compatibility is required for WIN32.
option(HIPCC_BACKWARD_COMPATIBILITY "Enable HIPCC backward compatibility" ON)
if(NOT WIN32)
  if(HIPCC_BACKWARD_COMPATIBILITY)
    include(hipcc-backward-compat.cmake)
  endif()
endif()

if(NOT ROCM_DEP_ROCMCORE)
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_AMD_PACKAGE_DEPENDS ${CPACK_DEBIAN_AMD_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_NVIDIA_PACKAGE_DEPENDS ${CPACK_DEBIAN_NVIDIA_PACKAGE_DEPENDS})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_AMD_PACKAGE_REQUIRES ${CPACK_RPM_AMD_PACKAGE_REQUIRES})
  string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_NVIDIA_PACKAGE_REQUIRES ${CPACK_RPM_NVIDIA_PACKAGE_REQUIRES})
endif()

# Static packaging
if(NOT BUILD_SHARED_LIBS)
  # For static builds change the pakage name
  set(CPACK_DEBIAN_AMD_PACKAGE_NAME "hipcc-static-dev")
  set(CPACK_RPM_AMD_PACKAGE_NAME "hipcc-static-devel")
endif()

include(CPack)
