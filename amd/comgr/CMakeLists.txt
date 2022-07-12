cmake_minimum_required(VERSION 3.13.4)

project(amd_comgr VERSION "2.4.0" LANGUAGES C CXX)
set(amd_comgr_NAME "${PROJECT_NAME}")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(FILE_REORG_BACKWARD_COMPATIBILITY "Enable File Reorg with backward compatibility" ON)
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
  src/comgr-compiler.cpp
  src/comgr.cpp
  src/comgr-device-libs.cpp
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
message("")

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

if (UNIX)
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS
    -fno-rtti -Wall -Wno-attributes -fms-extensions -fvisibility=hidden)
  # TODO: Confirm this is actually needed due to LLVM/Clang code
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_OPTIONS -fno-strict-aliasing)
  list(APPEND AMD_COMGR_PRIVATE_COMPILE_DEFINITIONS
    _GNU_SOURCE __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS)
  list(APPEND AMD_COMGR_PUBLIC_LINKER_OPTIONS -pthread)
  if (NOT APPLE AND COMGR_BUILD_SHARED_LIBS)
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/src/exportmap.in
      ${CMAKE_CURRENT_BINARY_DIR}/src/exportmap @ONLY)
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

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/amd_comgr.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/amd_comgr.h @ONLY)

include(bc2h)
include(opencl_pch)
include(DeviceLibs)

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

if (NOT COMGR_BUILD_SHARED_LIBS)
  set_target_properties(amd_comgr PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

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

set(AMD_COMGR_CONFIG_NAME amd_comgr-config.cmake)
set(AMD_COMGR_TARGETS_NAME amd_comgr-targets.cmake)
set(AMD_COMGR_VERSION_NAME amd_comgr-config-version.cmake)
set(AMD_COMGR_PACKAGE_PREFIX ${CMAKE_INSTALL_LIBDIR}/cmake/amd_comgr)

# Generate the build-tree package.
set(AMD_COMGR_PREFIX_CODE)
if (NOT COMGR_BUILD_SHARED_LIBS)
  string(APPEND AMD_COMGR_PREFIX_CODE "\ninclude(CMakeFindDependencyMacro)\n")
  string(APPEND AMD_COMGR_PREFIX_CODE "find_dependency(Clang REQUIRED)\n")
  string(APPEND AMD_COMGR_PREFIX_CODE "find_dependency(LLD REQUIRED)\n")
endif()

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
  COMPONENT amd-comgr
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/amd_comgr.h"
  COMPONENT amd-comgr
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${amd_comgr_NAME})

#File reorg Backward compatibility function
if(FILE_REORG_BACKWARD_COMPATIBILITY)
  include(comgr-backward-compat.cmake)
endif()

install(FILES
  "README.md"
  "LICENSE.txt"
  "NOTICES.txt"
  COMPONENT amd-comgr
  DESTINATION ${CMAKE_INSTALL_DOCDIR})

# Generate the install-tree package.
set(AMD_COMGR_PREFIX_CODE "
# Derive absolute install prefix from config file path.
get_filename_component(AMD_COMGR_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
string(REGEX REPLACE "/" ";" count "${AMD_COMGR_PACKAGE_PREFIX}")
foreach(p ${count})
  set(AMD_COMGR_PREFIX_CODE "${AMD_COMGR_PREFIX_CODE}
get_filename_component(AMD_COMGR_PREFIX \"\${AMD_COMGR_PREFIX}\" PATH)")
endforeach()

if (NOT COMGR_BUILD_SHARED_LIBS)
  string(APPEND AMD_COMGR_PREFIX_CODE "\ninclude(CMakeFindDependencyMacro)\n")
  string(APPEND AMD_COMGR_PREFIX_CODE "find_dependency(Clang REQUIRED)\n")
  string(APPEND AMD_COMGR_PREFIX_CODE "find_dependency(LLD REQUIRED)\n")
endif()

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

if(TARGET clangFrontendTool)
  set(CLANG_LIBS
    clangFrontendTool)
else()
  set(CLANG_LIBS
    clang-cpp)
endif()

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

include(CTest)
if(BUILD_TESTING)
  add_custom_target(check-comgr COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS amd_comgr)
  if (NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set_property(GLOBAL APPEND PROPERTY LLVM_ADDITIONAL_TEST_TARGETS check-comgr)
  endif()
  add_subdirectory(test)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  # Add packaging directives for amd_comgr
  set(CPACK_PACKAGE_NAME comgr)
  set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
  set(CPACK_PACKAGE_DESCRIPTION "Library to provide support functions\nfor ROCm code objects.")
  set(CPACK_PACKAGE_VERSION_MAJOR "${amd_comgr_VERSION_MAJOR}")
  set(CPACK_PACKAGE_VERSION_MINOR "${amd_comgr_VERSION_MINOR}")
  set(CPACK_PACKAGE_VERSION_PATCH "${amd_comgr_VERSION_PATCH}")
  set(CPACK_PACKAGE_CONTACT "ROCm Compiler Support <rocm.compiler.support@amd.com>")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

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

  # Debian package specific variables
  set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/RadeonOpenCompute/ROCm-CompilerSupport")
  if (LLVM_LINK_LLVM_DYLIB)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libtinfo-dev, rocm-core, rocm-llvm-core")
  else()
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libtinfo-dev, rocm-core")
  endif()
  if (DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE $ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
  else()
    set(CPACK_DEBIAN_PACKAGE_RELEASE "local")
  endif()

  # RPM package specific variables
  if (LLVM_LINK_LLVM_DYLIB)
    set(CPACK_RPM_PACKAGE_REQUIRES "ncurses, rocm-core, rocm-llvm-core")
  else()
    set(CPACK_RPM_PACKAGE_REQUIRES "ncurses, rocm-core")
  endif()
  if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "${CPACK_PACKAGING_INSTALL_PREFIX}")
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
      string(REGEX REPLACE ",? ?rocm-core" "" CPACK_DEBIAN_PACKAGE_DEPENDS ${CPACK_DEBIAN_PACKAGE_DEPENDS})
  endif()

  include(CPack)
endif()
