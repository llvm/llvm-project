cmake_minimum_required(VERSION 3.2.0)

# Build ROCM-Compiler-Support with ccache if the package is present.
set(ROCM_COMPILER_SUPPORT_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if(ROCM_COMPILER_SUPPORT_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(FATAL_ERROR "Unable to find the program ccache. Set ROCM_COMPILER_SUPPORT_CCACHE_BUILD to OFF")
  endif()
endif()

project(code_object_manager)

find_package(Clang REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

find_package(AMDDeviceLibs CONFIG)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# FIXME: Remove when yaml-cpp is removed from the build.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

if (ENABLE_ASAN)
  set(ASAN_LINKER_FLAGS "-fsanitize=address")
  set(ASAN_COMPILER_FLAGS "-fno-omit-frame-pointer ${ASAN_LINKER_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_COMPILER_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_COMPILER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_LINKER_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_LINKER_FLAGS}")
endif()

set(AMD_COMGR_PRIVATE_COMPILE_OPTIONS)
set(AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS)
set(AMD_COMGR_PUBLIC_LINKER_OPTIONS)
set(AMD_COMGR_PRIVATE_LINKER_OPTIONS)

if (UNIX)
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS
    -fno-rtti -Wall -Wno-attributes -fms-extensions -fvisibility=hidden)
  # TODO: Confirm this is actually needed due to LLVM/Clang code
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS -fno-strict-aliasing)
  if (CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS -g)
  endif()
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS
    _GNU_SOURCE __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS)
  list(APPEND AMD_COMGR_PUBLIC_LINKER_OPTIONS -pthread)
  if (NOT APPLE)
    list(APPEND AMD_COMGR_PRIVATE_LINKER_OPTIONS
      "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exportmap")
    # When building a shared library with -fsanitize=address we can't be
    # strict about undefined symbol references, as Clang won't include
    # libasan in the link, see
    # https://clang.llvm.org/docs/AddressSanitizer.html
    if (NOT ENABLE_ASAN)
      list(APPEND AMD_COMGR_PRIVATE_LINKER_OPTIONS
        -Wl,--no-undefined)
    endif()
  endif()
else()
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS "/W0")
endif()

# Windows is strict about visibility of exports in shared libraries, so we ask
# GCC/Clang to also be strict, and then explicitly mark each exported symbol in
# the shared header.
list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS AMD_EXPORT)

file(GLOB SOURCES "src/*.cpp")
if (NOT AMDDeviceLibs_FOUND)
  file(GLOB DEVICE_LIBS_SRC "src/comgr-device-libs.cpp")
  list(REMOVE_ITEM SOURCES ${DEVICE_LIBS_SRC})
endif()
add_library(amd_comgr SHARED ${SOURCES})

if (AMDDeviceLibs_FOUND)
  include(bc2h)
  include(opencl_pch)
  include(DeviceLibs)
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS DEVICE_LIBS)
endif()

set_target_properties(amd_comgr PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED Yes
  CXX_EXTENSIONS No)
target_compile_options(amd_comgr
  PRIVATE "${AMD_COMGR_PRIVATE_COMPILE_OPTIONS}")
target_compile_definitions(amd_comgr
  PRIVATE "${AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS}")
target_include_directories(amd_comgr
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

set(AMD_COMGR_CONFIG_NAME amd_comgr-config.cmake)
set(AMD_COMGR_TARGETS_NAME amd_comgr-targets.cmake)
set(AMD_COMGR_PACKAGE_PREFIX lib/cmake/amd_comgr)

# Generate the build-tree package.
set(AMD_COMGR_PREFIX_CODE)
set(AMD_COMGR_TARGETS_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_TARGETS_NAME}")
export(TARGETS amd_comgr
  FILE "${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_TARGETS_NAME}")
configure_file("cmake/${AMD_COMGR_CONFIG_NAME}.in"
  "${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_CONFIG_NAME}"
  @ONLY)

install(TARGETS amd_comgr
  EXPORT amd_comgr_export
  DESTINATION lib)

install(FILES
  "include/amd_comgr.h"
  DESTINATION include)

install(FILES
  "README.md"
  "LICENSE.txt"
  "NOTICES.txt"
  DESTINATION share/amd_comgr)

# Generate the install-tree package.
set(AMD_COMGR_PREFIX_CODE "
# Derive absolute install prefix from config file path.
get_filename_component(AMD_COMGR_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
string(REGEX REPLACE "/" ";" count "${AMD_COMGR_PACKAGE_PREFIX}")
foreach(p ${count})
  set(AMD_COMGR_PREFIX_CODE "${AMD_COMGR_PREFIX_CODE}
get_filename_component(AMD_COMGR_PREFIX \"\${AMD_COMGR_PREFIX}\" PATH)")
endforeach()
set(AMD_COMGR_TARGETS_PATH "\${AMD_COMGR_PREFIX}/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_TARGETS_NAME}")
configure_file("cmake/${AMD_COMGR_CONFIG_NAME}.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_COMGR_CONFIG_NAME}.install"
  @ONLY)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_COMGR_CONFIG_NAME}.install"
  DESTINATION "${AMD_COMGR_PACKAGE_PREFIX}"
  RENAME "${AMD_COMGR_CONFIG_NAME}")
install(EXPORT amd_comgr_export
  DESTINATION "${AMD_COMGR_PACKAGE_PREFIX}"
  FILE "${AMD_COMGR_TARGETS_NAME}")

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Enable testing" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Enable parse tools" FORCE)
add_subdirectory(yaml-cpp EXCLUDE_FROM_ALL)
include_directories(./yaml-cpp/include)

set(CLANG_LIBS
  clangFrontendTool)

set(LLD_LIBS
  lldELF
  lldCommon)

llvm_map_components_to_libnames(LLVM_LIBS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  Symbolize)

target_link_libraries(amd_comgr
  PUBLIC
    ${AMD_COMGR_PUBLIC_LINKER_OPTIONS}
  PRIVATE
    ${AMD_COMGR_PRIVATE_LINKER_OPTIONS}
    yaml-cpp
    ${CLANG_LIBS}
    ${LLD_LIBS}
    ${LLVM_LIBS}
)

if (NOT UNIX)
  target_link_libraries(amd_comgr
    PRIVATE version)
endif()

enable_testing()
add_subdirectory(test)

# Add packaging directives for amd_comgr
set(CPACK_PACKAGE_NAME comgr)
set(CPACK_PACKAGE_VENDOR "AMD")
set(CPACK_PACKAGE_VERSION_MAJOR 1)
set(CPACK_PACKAGE_VERSION_MINOR 2)
set(CPACK_PACKAGE_VERSION_PATCH 0)
set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Library to provide support functions")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

# Debian package specific variables
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE
  "https://github.com/RadeonOpenCompute/ROCm-CompilerSupport")

# RPM package specific variables
if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "${CPACK_PACKAGING_INSTALL_PREFIX}")
endif()

if (NOT CPack_CMake_INCLUDED)
  include(CPack)
endif()
