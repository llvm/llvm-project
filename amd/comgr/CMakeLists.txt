cmake_minimum_required(VERSION 3.13.4)

include(CheckIncludeFileCXX)

file(READ "VERSION.txt" comgr_ver_file)

string(REGEX MATCH "#COMGR_VERSION_MAJOR\n([0-9]*)" _ ${comgr_ver_file})
set (ver_major ${CMAKE_MATCH_1})
string(REGEX MATCH "#COMGR_VERSION_MINOR\n([0-9]*)" _ ${comgr_ver_file})
set (ver_minor ${CMAKE_MATCH_1})

message("Comgr Version: ${ver_major}.${ver_minor}.0")

project(amd_comgr VERSION "${ver_major}.${ver_minor}.0" LANGUAGES C CXX)
set(amd_comgr_NAME "${PROJECT_NAME}")

# Get git branch and commit hash to add to log for easier debugging.
# Modeled after https://github.com/pmirshad/cmake-with-git-metadata.
# Current working branch
if(EXISTS "${CMAKE_SOURCE_DIR}/../../.git")
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE AMD_COMGR_GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Shortened commit hash
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE AMD_COMGR_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
else(EXISTS "${CMAKE_SOURCE_DIR}/../../.git")
  set(AMD_COMGR_GIT_BRANCH "not-available")
  set(AMD_COMGR_GIT_COMMIT "not-available")
endif(EXISTS "${CMAKE_SOURCE_DIR}/../../.git")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorg with backward compatibility" OFF)
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

find_package(ROCM QUIET)
if (ROCM_FOUND)
  include(ROCMSetupVersion)
  rocm_setup_version(VERSION "${amd_comgr_VERSION}")
endif()

# BUILD_SHARED_LIBS is a frustratingly global variable common to all
# projects. LLVM also defines an option for the same varible with the
# opposite default, which will overwrite our default preference
# here. Ignore the regular BUILD_SHARED_LIBS in an embedded llvm
# build. Try to use BUILD_SHARED_LIBS to hint our project specific
# version in a standalone build.
set(build_shared_libs_default ON)
if(NOT DEFINED LLVM_SOURCE_DIR AND DEFINED BUILD_SHARED_LIBS)
  set(build_shared_libs_default ${BUILD_SHARED_LIBS})
endif()

option(COMGR_BUILD_SHARED_LIBS "Build the shared library"
       ${build_shared_libs_default})

set(SOURCES
  src/comgr-cache.cpp
  src/comgr-cache-command.cpp
  src/comgr-cache-bundler-command.cpp
  src/comgr-compiler.cpp
  src/comgr.cpp
  src/comgr-device-libs.cpp
  src/comgr-diagnostic-handler.cpp
  src/comgr-disassembly.cpp
  src/comgr-elfdump.cpp
  src/comgr-env.cpp
  src/comgr-metadata.cpp
  src/comgr-objdump.cpp
  src/comgr-signal.cpp
  src/comgr-symbol.cpp
  src/comgr-symbolizer.cpp
  src/time-stat/time-stat.cpp)

if(COMGR_BUILD_SHARED_LIBS)
  add_library(amd_comgr SHARED ${SOURCES})
  # Windows doesn't have a strip utility, so CMAKE_STRIP won't be set.
  if((CMAKE_BUILD_TYPE STREQUAL "Release") AND NOT ("${CMAKE_STRIP}" STREQUAL ""))
    if (APPLE)
      # Building on Mac fails unless -x is passed to the strip command
      add_custom_command(TARGET amd_comgr POST_BUILD COMMAND ${CMAKE_STRIP} -x $<TARGET_FILE:amd_comgr>)
    else()
      add_custom_command(TARGET amd_comgr POST_BUILD COMMAND ${CMAKE_STRIP} $<TARGET_FILE:amd_comgr>)
    endif()
  endif()
else()
  add_library(amd_comgr STATIC ${SOURCES})
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(AMDDeviceLibs REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)
  find_package(LLD REQUIRED CONFIG)

  target_include_directories(amd_comgr
    PRIVATE
      ${LLVM_INCLUDE_DIRS}
      ${CLANG_INCLUDE_DIRS}
      ${LLD_INCLUDE_DIRS})
else()
  # If building with LLVM_EXTERNAL_PROJECTS, we've already picked up
  # the include directories for LLVM, but not clang.
  #
  if (LLVM_EXTERNAL_CLANG_SOURCE_DIR)
    target_include_directories(amd_comgr
      PRIVATE
        ${LLVM_EXTERNAL_CLANG_SOURCE_DIR}/include
        ${LLVM_BINARY_DIR}/tools/clang/include)
  endif()

  if (LLVM_EXTERNAL_LLD_SOURCE_DIR)
    target_include_directories(amd_comgr
      PRIVATE
        ${LLVM_EXTERNAL_LLD_SOURCE_DIR}/include
        ${LLVM_BINARY_DIR}/tools/lld/include)
  endif()
endif()

target_include_directories(amd_comgr
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

message("")
message("------------LLVM_DIR: ${LLVM_DIR}")
message("---LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message("---LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS}")
message("-----------Clang_DIR: ${Clang_DIR}")
message("--CLANG_INCLUDE_DIRS: ${CLANG_INCLUDE_DIRS}")
message("----LLD_INCLUDE_DIRS: ${LLD_INCLUDE_DIRS}")
message("---AMDDeviceLibs_DIR: ${AMDDeviceLibs_DIR}")
message("------------ROCM_DIR: ${ROCM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (ADDRESS_SANITIZER)
  set(ASAN_LINKER_FLAGS "-fsanitize=address")
  set(ASAN_COMPILER_FLAGS "-fno-omit-frame-pointer -fsanitize=address")

  if (NOT CMAKE_COMPILER_IS_GNUCC)
    if (COMGR_BUILD_SHARED_LIBS)
      set(ASAN_LINKER_FLAGS "${ASAN_LINKER_FLAGS} -shared-libsan")
    else()
      set(ASAN_LINKER_FLAGS "${ASAN_LINKER_FLAGS} -static-libsan")
    endif()
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ASAN_COMPILER_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_COMPILER_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ASAN_LINKER_FLAGS} -s")
  set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} ${ASAN_LINKER_FLAGS}")
endif()

set(AMD_COMGR_PRIVATE_COMPILE_OPTIONS)
set(AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS ${LLVM_DEFINITIONS})
set(AMD_COMGR_PUBLIC_LINKER_OPTIONS)
set(AMD_COMGR_PRIVATE_LINKER_OPTIONS)

list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS "AMD_COMGR_GIT_COMMIT=${AMD_COMGR_GIT_COMMIT}")
list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS "AMD_COMGR_GIT_BRANCH=${AMD_COMGR_GIT_BRANCH}")
message("----COMGR_GIT_COMMIT: ${AMD_COMGR_GIT_COMMIT}")
message("----COMGR_GIT_BRANCH: ${AMD_COMGR_GIT_BRANCH}")
message("")

option(COMGR_DISABLE_SPIRV "To disable SPIRV in Comgr" OFF)

if (NOT COMGR_DISABLE_SPIRV)
  # TODO: Explore switching this to CHECK_INCLUDE_FILE_CXX() macro
  if (NOT EXISTS "${LLVM_INCLUDE_DIRS}/LLVMSPIRVLib/LLVMSPIRVLib.h")
    message("-- LLVMSPIRVLib/LLVMSPIRVLib.h not found")
    set(COMGR_DISABLE_SPIRV ON)
  else()
    message("-- LLVMSPIRVLib/LLVMSPIRVLib.h found")
  endif()
endif()

if(${COMGR_DISABLE_SPIRV})
    list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS "COMGR_DISABLE_SPIRV")
  message("-- Comgr SPIRV Disabled")
else()
  message("-- Comgr SPIRV Enabled")
endif()


if (UNIX)
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS
    -fno-rtti -Wall -Wno-attributes -fms-extensions -fvisibility=hidden)
  # TODO: Confirm this is actually needed due to LLVM/Clang code
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS -fno-strict-aliasing)
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS
    _GNU_SOURCE __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS AMD_COMGR_BUILD)
  list(APPEND AMD_COMGR_PUBLIC_LINKER_OPTIONS -pthread)
  if (NOT APPLE AND COMGR_BUILD_SHARED_LIBS)
    configure_file(
      src/exportmap.in
      src/exportmap @ONLY)
    list(APPEND AMD_COMGR_PRIVATE_LINKER_OPTIONS
      "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/src/exportmap")
    # When building a shared library with -fsanitize=address we can't be
    # strict about undefined symbol references, as Clang won't include
    # libasan in the link, see
    # https://clang.llvm.org/docs/AddressSanitizer.html
    if (NOT ADDRESS_SANITIZER)
      list(APPEND AMD_COMGR_PRIVATE_LINKER_OPTIONS
        -Wl,--no-undefined)
    endif()
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS
    "/wd4244" #[[Suppress 'argument' : conversion from 'type1' to 'type2', possible loss of data]]
    "/wd4624" #[[Suppress 'derived class' : destructor could not be generated because a base class destructor is inaccessible]]
    "/wd4267" #[[Suppress 'var' : conversion from 'size_t' to 'type', possible loss of data]]
    "/wd4291" #[[Suppress 'declaration' : no matching operator delete found; memory will not be freed if initialization throws an exception]]
    "/wd4146" #[[Suppress 'unary minus operator applied to unsigned type, result still unsigned]])
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS _HAS_EXCEPTIONS=0)
endif()

# Windows is strict about visibility of exports in shared libraries, so we ask
# GCC/Clang to also be strict, and then explicitly mark each exported symbol in
# the shared header.
list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS AMD_COMGR_EXPORT)

include(bc2h)
include(opencl_pch)
include(DeviceLibs)

# Add major version to the name on windows, including Win64
if (WIN32)
  set_target_properties(amd_comgr PROPERTIES
    OUTPUT_NAME "amd_comgr_${amd_comgr_VERSION_MAJOR}")
endif()

set_target_properties(amd_comgr PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED Yes
  CXX_EXTENSIONS No)
if (ROCM_FOUND)
  rocm_set_soversion(amd_comgr "${amd_comgr_VERSION_MAJOR}.${amd_comgr_VERSION_MINOR}")
else()
  set_target_properties(amd_comgr PROPERTIES
    SOVERSION "${amd_comgr_VERSION_MAJOR}"
    VERSION "${amd_comgr_VERSION_MAJOR}.${amd_comgr_VERSION_MINOR}.${amd_comgr_VERSION_PATCH}")
endif()

if (NOT COMGR_BUILD_SHARED_LIBS)
  set_target_properties(amd_comgr PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Overwrite the name on 32-bit Linux and Windows
if (CMAKE_SIZEOF_VOID_P EQUAL 4)
  set_target_properties(amd_comgr PROPERTIES OUTPUT_NAME "amd_comgr32")
endif()

target_compile_options(amd_comgr
  PRIVATE "${AMD_COMGR_PRIVATE_COMPILE_OPTIONS}")
target_compile_definitions(amd_comgr
  PRIVATE "${AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS}")
target_include_directories(amd_comgr
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

configure_file(
  include/amd_comgr.h.in
  include/amd_comgr.h @ONLY)

set(AMD_COMGR_CONFIG_NAME amd_comgr-config.cmake)
set(AMD_COMGR_TARGETS_NAME amd_comgr-targets.cmake)
set(AMD_COMGR_VERSION_NAME amd_comgr-config-version.cmake)
set(AMD_COMGR_PACKAGE_PREFIX cmake/amd_comgr)

# Generate the build-tree package.
set(AMD_COMGR_PREFIX_CODE)
if (NOT COMGR_BUILD_SHARED_LIBS)
  string(APPEND AMD_COMGR_PREFIX_CODE "\ninclude(CMakeFindDependencyMacro)\n")
  string(APPEND AMD_COMGR_PREFIX_CODE "find_dependency(Clang REQUIRED)\n")
  string(APPEND AMD_COMGR_PREFIX_CODE "find_dependency(LLD REQUIRED)\n")
endif()

set(AMD_COMGR_TARGETS_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/lib/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_TARGETS_NAME}")
set(AMD_COMGR_VERSION_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/lib/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_VERSION_NAME}")
export(TARGETS amd_comgr
  FILE "lib/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_TARGETS_NAME}")
configure_file("cmake/${AMD_COMGR_CONFIG_NAME}.in"
  "lib/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_CONFIG_NAME}"
  @ONLY)
write_basic_package_version_file("${AMD_COMGR_VERSION_PATH}"
  VERSION "${amd_comgr_VERSION}"
  COMPATIBILITY SameMajorVersion)

if(ENABLE_ASAN_PACKAGING)
  install(TARGETS amd_comgr
    EXPORT amd_comgr_export
    COMPONENT asan
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
  install(TARGETS amd_comgr
    EXPORT amd_comgr_export
    COMPONENT amd-comgr
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/amd_comgr.h"
  COMPONENT amd-comgr
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${amd_comgr_NAME})

#File reorg Backward compatibility function
if(FILE_REORG_BACKWARD_COMPATIBILITY)
# To enable/disable #error in wrapper header files
 if(NOT DEFINED ROCM_HEADER_WRAPPER_WERROR)
      if(DEFINED ENV{ROCM_HEADER_WRAPPER_WERROR})
          set(ROCM_HEADER_WRAPPER_WERROR "$ENV{ROCM_HEADER_WRAPPER_WERROR}"
                  CACHE STRING "Header wrapper warnings as errors.")
      else()
          set(ROCM_HEADER_WRAPPER_WERROR "OFF" CACHE STRING "Header wrapper warnings as errors.")
      endif()
  endif()
  if(ROCM_HEADER_WRAPPER_WERROR)
      set(deprecated_error 1)
  else()
      set(deprecated_error 0)
  endif()

  include(comgr-backward-compat.cmake)
endif()

if(ENABLE_ASAN_PACKAGING)
  install(FILES
    "LICENSE.txt"
    COMPONENT asan
    DESTINATION ${CMAKE_INSTALL_DOCDIR}-asan)
else()
  install(FILES
    "README.md"
    "LICENSE.txt"
    "NOTICES.txt"
    COMPONENT amd-comgr
    DESTINATION ${CMAKE_INSTALL_DOCDIR})
endif()

# Generate the install-tree package.
set(AMD_COMGR_PREFIX_CODE "
# Derive absolute install prefix from config file path.
get_filename_component(AMD_COMGR_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
string(REGEX REPLACE "/" ";" count "${CMAKE_INSTALL_LIBDIR}/${AMD_COMGR_PACKAGE_PREFIX}")
foreach(p ${count})
  set(AMD_COMGR_PREFIX_CODE "${AMD_COMGR_PREFIX_CODE}
get_filename_component(AMD_COMGR_PREFIX \"\${AMD_COMGR_PREFIX}\" PATH)")
endforeach()

if (NOT COMGR_BUILD_SHARED_LIBS)
  string(APPEND AMD_COMGR_PREFIX_CODE "\ninclude(CMakeFindDependencyMacro)\n")
  string(APPEND AMD_COMGR_PREFIX_CODE "find_dependency(Clang REQUIRED)\n")
  string(APPEND AMD_COMGR_PREFIX_CODE "find_dependency(LLD REQUIRED)\n")
endif()

set(AMD_COMGR_TARGETS_PATH "\${AMD_COMGR_PREFIX}/${CMAKE_INSTALL_LIBDIR}/${AMD_COMGR_PACKAGE_PREFIX}/${AMD_COMGR_TARGETS_NAME}")
configure_file("cmake/${AMD_COMGR_CONFIG_NAME}.in"
  "${AMD_COMGR_CONFIG_NAME}.install"
  @ONLY)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${AMD_COMGR_CONFIG_NAME}.install"
  COMPONENT amd-comgr
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/${AMD_COMGR_PACKAGE_PREFIX}"
  RENAME "${AMD_COMGR_CONFIG_NAME}")
install(EXPORT amd_comgr_export
  COMPONENT amd-comgr
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/${AMD_COMGR_PACKAGE_PREFIX}"
  FILE "${AMD_COMGR_TARGETS_NAME}")
install(FILES
  "${AMD_COMGR_VERSION_PATH}"
  COMPONENT amd-comgr
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/${AMD_COMGR_PACKAGE_PREFIX}")

if(TARGET clangFrontendTool)
  set(CLANG_LIBS
    clangBasic
    clangDriver
    clangSerialization
    clangFrontend
    clangFrontendTool)
else()
  set(CLANG_LIBS
    clang-cpp)
endif()

set(LLD_LIBS
  lldELF
  lldCommon)

if (${COMGR_DISABLE_SPIRV})
  set(SPIRV_DYNAMIC_LIB "")
  set(SPIRV_STATIC_LIB "")
else()
  set(SPIRV_DYNAMIC_LIB "LLVMSPIRVAMDLib")
  set(SPIRV_STATIC_LIB "SPIRVAMDLib")
endif()

if (LLVM_LINK_LLVM_DYLIB)
  set(LLVM_LIBS LLVM ${SPIRV_DYNAMIC_LIB})
else()
  llvm_map_components_to_libnames(LLVM_LIBS
    ${LLVM_TARGETS_TO_BUILD}
    BinaryFormat
    BitReader
    BitWriter
    CodeGen
    Core
    DebugInfoDWARF
    Demangle
    IRReader
    Linker
    MC
    MCDisassembler
    MCParser
    Object
    Option
    Support
    Symbolize
    TargetParser
    ${SPIRV_STATIC_LIB}
    )
endif()

target_link_options(amd_comgr
  PUBLIC
    ${AMD_COMGR_PUBLIC_LINKER_OPTIONS}
  PRIVATE
    ${AMD_COMGR_PRIVATE_LINKER_OPTIONS})

target_link_libraries(amd_comgr
  PRIVATE
    ${LLD_LIBS}
    ${LLVM_LIBS}
    ${CLANG_LIBS})

if (NOT UNIX)
  target_link_libraries(amd_comgr
    PRIVATE version)
endif()

find_package(Threads)
target_link_libraries(amd_comgr PRIVATE ${CMAKE_THREAD_LIBS_INIT})

find_library(LIBRT rt)
if(LIBRT)
  target_link_libraries(amd_comgr PRIVATE ${LIBRT})
endif()


if (NOT WIN32)
  target_link_libraries(amd_comgr
    PRIVATE
      c
      ${CMAKE_DL_LIBS})
endif()

include(CTest)
if(BUILD_TESTING)
  add_custom_target(check-comgr COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS amd_comgr)
  if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set_property(GLOBAL APPEND PROPERTY LLVM_ADDITIONAL_TEST_TARGETS check-comgr)
  endif()
  add_subdirectory(test)
  add_subdirectory(test-lit)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Add packaging directives for amd_comgr
  if(ENABLE_ASAN_PACKAGING)
  # Only libraries required for ASAN Package
    set(CPACK_COMPONENTS_ALL asan)
    set(PKG_DESC_SUMMARY "AddressSanitizer Instrumented Libraries to provide support functions for ROCm code objects.")
  elseif(NOT COMGR_BUILD_SHARED_LIBS)
    set(CPACK_COMPONENTS_ALL amd-comgr)
    set(PKG_DESC_SUMMARY "Static Library to provide support functions for ROCm code objects.")
  else()
    set(CPACK_COMPONENTS_ALL amd-comgr)
    set(PKG_DESC_SUMMARY "Library to provide support functions for ROCm code objects.")
  endif()
  set(CPACK_PACKAGE_NAME comgr)
  set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PKG_DESC_SUMMARY})
  set(CPACK_PACKAGE_DESCRIPTION "This package contains the AMD ${CPACK_PACKAGE_DESCRIPTION_SUMMARY}.")
  set(CPACK_PACKAGE_VERSION_MAJOR "${amd_comgr_VERSION_MAJOR}")
  set(CPACK_PACKAGE_VERSION_MINOR "${amd_comgr_VERSION_MINOR}")
  set(CPACK_PACKAGE_VERSION_PATCH "${amd_comgr_VERSION_PATCH}")
  set(CPACK_PACKAGE_CONTACT "ROCm Compiler Support <rocm.compiler.support@amd.com>")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

  # ASAN Specific variables
  set(CPACK_DEBIAN_ASAN_PACKAGE_NAME comgr-asan)
  set(CPACK_RPM_ASAN_PACKAGE_NAME comgr-asan)

 # Make proper version for appending
  set(ROCM_VERSION_FOR_PACKAGE "")
  if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
    set(ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_LIBPATCH_VERSION})
  elseif(DEFINED ENV{ROCM_VERSION})
    string(REGEX REPLACE "." "" ROCM_VERSION_FOR_PACKAGE $ENV{ROCM_VERSION})
  else()
    # Default Case, set to 99999
    set(ROCM_VERSION_FOR_PACKAGE "99999")
  endif()

  # Archive package specific variable
  set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)

  # Debian package specific variables
  set(CPACK_DEB_COMPONENT_INSTALL ON)
  if(COMGR_BUILD_SHARED_LIBS)
    set(CPACK_DEBIAN_AMD-COMGR_PACKAGE_NAME comgr)
  else()
    set(CPACK_DEBIAN_AMD-COMGR_PACKAGE_NAME comgr-static-dev)
  endif()
  set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr")
  set(DEBIAN_DEPENDENCIES "libzstd1, zlib1g, libc6, libstdc++6, libgcc-s1 | libgcc1")
  if (LLVM_LINK_LLVM_DYLIB)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libtinfo-dev, rocm-core, rocm-llvm-core, ${DEBIAN_DEPENDENCIES}")
    set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "libtinfo-dev, rocm-core-asan, rocm-llvm-core, ${DEBIAN_DEPENDENCIES}")
  else()
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libtinfo-dev, rocm-core, ${DEBIAN_DEPENDENCIES}")
    set(CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS "libtinfo-dev, rocm-core-asan, ${DEBIAN_DEPENDENCIES}")
  endif()
  if (DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  else()
    set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
  endif()

  # RPM package specific variables
  set(CPACK_RPM_COMPONENT_INSTALL ON)
  if(COMGR_BUILD_SHARED_LIBS)
    set(CPACK_RPM_AMD-COMGR_PACKAGE_NAME comgr)
  else()
    set(CPACK_RPM_AMD-COMGR_PACKAGE_NAME comgr-static-devel)
  endif()

  execute_process(COMMAND rpm --eval %{?dist}
                 RESULT_VARIABLE PROC_RESULT
                 OUTPUT_VARIABLE EVAL_RESULT
                 OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(PROC_RESULT EQUAL "0" AND "${EVAL_RESULT}" STREQUAL ".el7")
    # In Centos using parentheses is causing cpack errors.
    # Set the dependencies specifically for centos
    set(RPM_DEPENDENCIES "zlib, glibc, libstdc++, libgcc")
  else()
    set(RPM_DEPENDENCIES "(zlib or libz1), (libzstd or libzstd1), glibc, (libstdc++ or libstdc++6), (libgcc or libgcc_s1)")
  endif()

  if (LLVM_LINK_LLVM_DYLIB)
    set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core, rocm-llvm-core, ${RPM_DEPENDENCIES}")
    set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan, rocm-llvm-core, ${RPM_DEPENDENCIES}")
  else()
    set(CPACK_RPM_PACKAGE_REQUIRES "rocm-core, ${RPM_DEPENDENCIES}")
    set(CPACK_RPM_ASAN_PACKAGE_REQUIRES "rocm-core-asan, ${RPM_DEPENDENCIES}")
  endif()
  if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE $ENV{CPACK_RPM_PACKAGE_RELEASE})
  else()
    set(CPACK_RPM_PACKAGE_RELEASE "local")
  endif()
  set(CPACK_RPM_PACKAGE_LICENSE "NCSA")

  # Get rpm distro
  if(CPACK_RPM_PACKAGE_RELEASE)
    set(CPACK_RPM_PACKAGE_RELEASE_DIST ON)
  endif()

  # Prepare final version for the CPACK use
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}.${ROCM_VERSION_FOR_PACKAGE}")

  # Set the names now using CPACK utility
  set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")
  set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
  # Remove dependency on rocm-core if -DROCM_DEP_ROCMCORE=ON not given to cmake
  if(NOT ROCM_DEP_ROCMCORE)
      string(REGEX REPLACE ",? ?rocm-core" "" CPACK_RPM_PACKAGE_REQUIRES ${CPACK_RPM_PACKAGE_REQUIRES})
      string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_RPM_ASAN_PACKAGE_REQUIRES ${CPACK_RPM_ASAN_PACKAGE_REQUIRES})
      string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
      string(REGEX REPLACE ",? ?rocm-core-asan" "" CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_ASAN_PACKAGE_DEPENDS})
  endif()

  include(CPack)
endif()
