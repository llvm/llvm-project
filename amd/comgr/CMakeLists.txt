cmake_minimum_required(VERSION 3.2.0)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Optionally, build Compiler Support with ccache.
set(ROCM_CCACHE_BUILD OFF CACHE BOOL "Set to ON for a ccache enabled build")
if (ROCM_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if (CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(WARNING "Unable to find ccache. Falling back to real compiler")
  endif() # if (CCACHE_PROGRAM)
endif() # if (ROCM_CCACHE_BUILD)

project(amd_comgr VERSION "1.6.0" LANGUAGES C CXX)

find_package(ROCM PATHS "/opt/rocm")
if (ROCM_FOUND)
  include(ROCMSetupVersion)
  rocm_setup_version(VERSION "${amd_comgr_VERSION}")
endif()

find_package(Clang REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})

# FIXME: There is no way to directly determine the include paths for LLD when
# working with an LLVM build tree, but we want to avoid having to install LLVM
# to build Comgr. This should eventually be fixed in usptream LLD so this can
# be removed.
if(DEFINED LLVM_BUILD_MAIN_SRC_DIR)
  set(INTERNAL_LLD_INCLUDE_DIRS "${LLVM_BUILD_MAIN_SRC_DIR}/tools/lld/include")
  set(EXTERNAL_LLD_INCLUDE_DIRS "${LLVM_BUILD_MAIN_SRC_DIR}/../lld/include")
  if (EXISTS "${INTERNAL_LLD_INCLUDE_DIRS}"
      AND IS_DIRECTORY "${INTERNAL_LLD_INCLUDE_DIRS}")
    set(LLD_INCLUDE_DIRS "${INTERNAL_LLD_INCLUDE_DIRS}")
  elseif (EXISTS "${EXTERNAL_LLD_INCLUDE_DIRS}"
      AND IS_DIRECTORY "${EXTERNAL_LLD_INCLUDE_DIRS}")
    set(LLD_INCLUDE_DIRS "${EXTERNAL_LLD_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "You are attempting to compile using an LLVM build tree, but the LLD include directory could not be located. The paths '${INTERNAL_LLD_INCLUDE_DIRS}' and '${EXTERNAL_LLD_INCLUDE_DIRS}' were tried.")
  endif()
  include_directories(${LLD_INCLUDE_DIRS})
endif()

find_package(AMDDeviceLibs CONFIG)
find_package(ZLIB)

message("")
message("------------LLVM_DIR: ${LLVM_DIR}")
message("---LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message("---LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
message("-----------Clang_DIR: ${Clang_DIR}")
message("--CLANG_INCLUDE_DIRS: ${CLANG_INCLUDE_DIRS}")
message("----LLD_INCLUDE_DIRS: ${LLD_INCLUDE_DIRS}")
message("---AMDDeviceLibs_DIR: ${AMDDeviceLibs_DIR}")
message("------------ROCM_DIR: ${ROCM_DIR}")
message("")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS "/W0" "/wd4244")
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS _HAS_EXCEPTIONS=0)
endif()

# Windows is strict about visibility of exports in shared libraries, so we ask
# GCC/Clang to also be strict, and then explicitly mark each exported symbol in
# the shared header.
list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS AMD_EXPORT)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/amd_comgr.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/amd_comgr.h @ONLY)

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
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED Yes
  CXX_EXTENSIONS No)
if (ROCM_FOUND)
  rocm_set_soversion(amd_comgr "${amd_comgr_VERSION_MAJOR}.${amd_comgr_VERSION_MINOR}")
else()
  set_target_properties(amd_comgr PROPERTIES
    SOVERSION "${amd_comgr_VERSION_MAJOR}"
    VERSION "${amd_comgr_VERSION_MAJOR}.${amd_comgr_VERSION_MINOR}.${amd_comgr_VERSION_PATCH}")
endif()
target_compile_options(amd_comgr
  PRIVATE "${AMD_COMGR_PRIVATE_COMPILE_OPTIONS}")
target_compile_definitions(amd_comgr
  PRIVATE "${AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS}")
target_include_directories(amd_comgr
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

set(AMD_COMGR_CONFIG_NAME amd_comgr-config.cmake)
set(AMD_COMGR_TARGETS_NAME amd_comgr-targets.cmake)
set(AMD_COMGR_VERSION_NAME amd_comgr-config-version.cmake)
set(AMD_COMGR_PACKAGE_PREFIX lib/cmake/amd_comgr)

# Generate the build-tree package.
set(AMD_COMGR_PREFIX_CODE)
set(AMD_COMGR_TARGETS_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_TARGETS_NAME}")
set(AMD_COMGR_VERSION_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_VERSION_NAME}")
export(TARGETS amd_comgr
  FILE "${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_TARGETS_NAME}")
configure_file("cmake/${AMD_COMGR_CONFIG_NAME}.in"
  "${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_CONFIG_NAME}"
  @ONLY)
write_basic_package_version_file("${AMD_COMGR_VERSION_PATH}"
  VERSION "${amd_comgr_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(TARGETS amd_comgr
  EXPORT amd_comgr_export
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/amd_comgr.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES
  "README.md"
  "LICENSE.txt"
  "NOTICES.txt"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/amd_comgr)

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
install(FILES
  "${AMD_COMGR_VERSION_PATH}"
  DESTINATION "${AMD_COMGR_PACKAGE_PREFIX}")

set(CLANG_LIBS
  clangFrontendTool)

set(LLD_LIBS
  lldELF
  lldCommon)

if (LLVM_LINK_LLVM_DYLIB)
  set(LLVM_LIBS LLVM)
else()
  llvm_map_components_to_libnames(LLVM_LIBS
    ${LLVM_TARGETS_TO_BUILD}
    DebugInfoDWARF
    Symbolize)
endif()

target_link_libraries(amd_comgr
  PUBLIC
    ${AMD_COMGR_PUBLIC_LINKER_OPTIONS}
  PRIVATE
    ${AMD_COMGR_PRIVATE_LINKER_OPTIONS}
    ${CLANG_LIBS}
    ${LLD_LIBS}
    ${LLVM_LIBS}
    ZLIB::ZLIB
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
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Library to provide support functions")
set(CPACK_PACKAGE_VERSION "${amd_comgr_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${amd_comgr_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${amd_comgr_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${amd_comgr_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "Advanced Micro Devices Inc.")
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
