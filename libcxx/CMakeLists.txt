# See www/CMake.html for instructions on how to build libcxx with CMake.

#===============================================================================
# Setup Project
#===============================================================================
cmake_minimum_required(VERSION 3.4.3)

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # Set MACOSX_RPATH=YES by default
endif()
if(POLICY CMP0022)
  cmake_policy(SET CMP0022 NEW) # Required when interacting with LLVM and Clang
endif()
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(libcxx CXX C)

  set(PACKAGE_NAME libcxx)
  set(PACKAGE_VERSION 8.0.0svn)
  set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
  set(PACKAGE_BUGREPORT "llvm-bugs@lists.llvm.org")

  # Find the LLVM sources and simulate LLVM CMake options.
  include(HandleOutOfTreeLLVM)
endif()

if (LIBCXX_STANDALONE_BUILD)
  include(FindPythonInterp)
  if( NOT PYTHONINTERP_FOUND )
    message(WARNING "Failed to find python interpreter. "
                    "The libc++ test suite will be disabled.")
    set(LLVM_INCLUDE_TESTS OFF)
  endif()
endif()

# Require out of source build.
include(MacroEnsureOutOfSourceBuild)
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate
 build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )

if (MSVC)
  set(LIBCXX_TARGETING_MSVC ON)
else()
  set(LIBCXX_TARGETING_MSVC OFF)
endif()

#===============================================================================
# Setup CMake Options
#===============================================================================
include(CMakeDependentOption)
include(HandleCompilerRT)

# Basic options ---------------------------------------------------------------
option(LIBCXX_ENABLE_ASSERTIONS "Enable assertions independent of build mode." OFF)
option(LIBCXX_ENABLE_SHARED "Build libc++ as a shared library." ON)
option(LIBCXX_ENABLE_STATIC "Build libc++ as a static library." ON)
option(LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY "Build libc++experimental.a" ON)
set(ENABLE_FILESYSTEM_DEFAULT ${LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY})
if (WIN32)
  set(ENABLE_FILESYSTEM_DEFAULT OFF)
endif()
option(LIBCXX_ENABLE_FILESYSTEM "Build filesystem as part of libc++fs.a"
    ${ENABLE_FILESYSTEM_DEFAULT})
option(LIBCXX_INCLUDE_TESTS "Build the libc++ tests." ${LLVM_INCLUDE_TESTS})

# Benchmark options -----------------------------------------------------------
option(LIBCXX_INCLUDE_BENCHMARKS "Build the libc++ benchmarks and their dependancies" ON)
set(LIBCXX_BENCHMARK_NATIVE_STDLIB "" CACHE STRING
        "Build the benchmarks against the specified native STL.
         The value must be one of libc++/libstdc++")
set(LIBCXX_BENCHMARK_NATIVE_GCC_TOOLCHAIN "" CACHE STRING
    "Use alternate GCC toolchain when building the native benchmarks")

if (LIBCXX_BENCHMARK_NATIVE_STDLIB)
  if (NOT (LIBCXX_BENCHMARK_NATIVE_STDLIB STREQUAL "libc++"
        OR LIBCXX_BENCHMARK_NATIVE_STDLIB STREQUAL "libstdc++"))
    message(FATAL_ERROR "Invalid value for LIBCXX_BENCHMARK_NATIVE_STDLIB: "
            "'${LIBCXX_BENCHMARK_NATIVE_STDLIB}'")
  endif()
endif()

option(LIBCXX_INCLUDE_DOCS "Build the libc++ documentation." ${LLVM_INCLUDE_DOCS})
set(LIBCXX_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}" CACHE STRING
    "Define suffix of library directory name (32/64)")
option(LIBCXX_INSTALL_HEADERS "Install the libc++ headers." ON)
option(LIBCXX_INSTALL_LIBRARY "Install the libc++ library." ON)
cmake_dependent_option(LIBCXX_INSTALL_STATIC_LIBRARY
  "Install the static libc++ library." ON
  "LIBCXX_ENABLE_STATIC;LIBCXX_INSTALL_LIBRARY" OFF)
cmake_dependent_option(LIBCXX_INSTALL_SHARED_LIBRARY
  "Install the shared libc++ library." ON
  "LIBCXX_ENABLE_SHARED;LIBCXX_INSTALL_LIBRARY" OFF)
option(LIBCXX_INSTALL_SUPPORT_HEADERS "Install libc++ support headers." ON)
cmake_dependent_option(LIBCXX_INSTALL_EXPERIMENTAL_LIBRARY
        "Install libc++experimental.a" ON
        "LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY;LIBCXX_INSTALL_LIBRARY" OFF)
cmake_dependent_option(LIBCXX_INSTALL_FILESYSTEM_LIBRARY
        "Install libc++fs.a" ON
        "LIBCXX_ENABLE_FILESYSTEM;LIBCXX_INSTALL_LIBRARY" OFF)

if (FUCHSIA)
  set(DEFAULT_ABI_VERSION 2)
else()
  set(DEFAULT_ABI_VERSION 1)
endif()
set(LIBCXX_ABI_VERSION ${DEFAULT_ABI_VERSION} CACHE STRING "ABI version of libc++.")
option(LIBCXX_ABI_UNSTABLE "Unstable ABI of libc++." OFF)
option(LIBCXX_ABI_FORCE_ITANIUM "Ignore auto-detection and force use of the Itanium ABI.")
option(LIBCXX_ABI_FORCE_MICROSOFT "Ignore auto-detection and force use of the Microsoft ABI.")
set(LIBCXX_ABI_DEFINES "" CACHE STRING "A semicolon separated list of ABI macros to define in the site config header.")
option(LIBCXX_USE_COMPILER_RT "Use compiler-rt instead of libgcc" OFF)

if (NOT LIBCXX_ENABLE_SHARED AND NOT LIBCXX_ENABLE_STATIC)
  message(FATAL_ERROR "libc++ must be built as either a shared or static library.")
endif()

# ABI Library options ---------------------------------------------------------
set(LIBCXX_CXX_ABI "default" CACHE STRING
    "Specify C++ ABI library to use.")
set(CXXABIS none default libcxxabi libcxxrt libstdc++ libsupc++ vcruntime)
set_property(CACHE LIBCXX_CXX_ABI PROPERTY STRINGS ;${CXXABIS})

# Setup the default options if LIBCXX_CXX_ABI is not specified.
if (LIBCXX_CXX_ABI STREQUAL "default")
  find_path(
    LIBCXX_LIBCXXABI_INCLUDES_INTERNAL
    cxxabi.h
    PATHS ${LLVM_MAIN_SRC_DIR}/projects/libcxxabi/include
          ${LLVM_MAIN_SRC_DIR}/runtimes/libcxxabi/include
          ${LLVM_MAIN_SRC_DIR}/../libcxxabi/include
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
  )
  if (LIBCXX_TARGETING_MSVC)
    # FIXME: Figure out how to configure the ABI library on Windows.
    set(LIBCXX_CXX_ABI_LIBNAME "vcruntime")
  elseif ((NOT LIBCXX_STANDALONE_BUILD OR HAVE_LIBCXXABI) AND
          IS_DIRECTORY "${LIBCXX_LIBCXXABI_INCLUDES_INTERNAL}")
    set(LIBCXX_CXX_ABI_LIBNAME "libcxxabi")
    set(LIBCXX_CXX_ABI_INCLUDE_PATHS "${LIBCXX_LIBCXXABI_INCLUDES_INTERNAL}")
    set(LIBCXX_CXX_ABI_INTREE 1)
  elseif (APPLE)
    set(LIBCXX_CXX_ABI_LIBNAME "libcxxabi")
    set(LIBCXX_CXX_ABI_SYSTEM 1)
  elseif (${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
    set(LIBCXX_CXX_ABI_LIBNAME "libcxxrt")
    set(LIBCXX_CXX_ABI_INCLUDE_PATHS "/usr/include/c++/v1")
  else()
    set(LIBCXX_CXX_ABI_LIBNAME "default")
  endif()
else()
  set(LIBCXX_CXX_ABI_LIBNAME "${LIBCXX_CXX_ABI}")
endif()

# Use a static copy of the ABI library when linking libc++. This option
# cannot be used with LIBCXX_ENABLE_ABI_LINKER_SCRIPT.
option(LIBCXX_ENABLE_STATIC_ABI_LIBRARY "Statically link the ABI library" OFF)

cmake_dependent_option(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY
  "Statically link the ABI library to static library" ON
  "LIBCXX_ENABLE_STATIC_ABI_LIBRARY;LIBCXX_ENABLE_STATIC" OFF)

cmake_dependent_option(LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY
  "Statically link the ABI library to shared library" ON
  "LIBCXX_ENABLE_STATIC_ABI_LIBRARY;LIBCXX_ENABLE_STATIC" OFF)

# Generate and install a linker script inplace of libc++.so. The linker script
# will link libc++ to the correct ABI library. This option is on by default
# on UNIX platforms other than Apple unless 'LIBCXX_ENABLE_STATIC_ABI_LIBRARY'
# is on. This option is also disabled when the ABI library is not specified
# or is specified to be "none".
set(ENABLE_LINKER_SCRIPT_DEFAULT_VALUE OFF)
if (LLVM_HAVE_LINK_VERSION_SCRIPT AND NOT LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY
      AND NOT LIBCXX_CXX_ABI_LIBNAME STREQUAL "none"
      AND NOT LIBCXX_CXX_ABI_LIBNAME STREQUAL "default"
      AND PYTHONINTERP_FOUND
      AND LIBCXX_ENABLE_SHARED)
    set(ENABLE_LINKER_SCRIPT_DEFAULT_VALUE ON)
endif()

option(LIBCXX_ENABLE_ABI_LINKER_SCRIPT
      "Use and install a linker script for the given ABI library"
      ${ENABLE_LINKER_SCRIPT_DEFAULT_VALUE})

set(ENABLE_NEW_DELETE_DEFAULT ON)
if (LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS)
# FIXME: This option should default to off. Unfortunatly GCC 4.9 fails to link
# programs due to undefined references to new/delete in libc++abi so to work
# around this libc++abi currently defaults LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS
# to ON. Once the GCC bug has been worked around this option should be changed
# back to OFF.
  set(ENABLE_NEW_DELETE_DEFAULT ON)
endif()

option(LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS
    "Build libc++ with definitions for operator new/delete. This option can
    be used to disable the definitions when libc++abi is expected to provide
    them" ${ENABLE_NEW_DELETE_DEFAULT})

# Build libc++abi with libunwind. We need this option to determine whether to
# link with libunwind or libgcc_s while running the test cases.
option(LIBCXXABI_USE_LLVM_UNWINDER "Build and use the LLVM unwinder." OFF)
option(LIBCXXABI_ENABLE_STATIC_UNWINDER "Statically link the LLVM unwinder." OFF)

# Target options --------------------------------------------------------------
option(LIBCXX_BUILD_32_BITS "Build 32 bit libc++." ${LLVM_BUILD_32_BITS})
set(LIBCXX_SYSROOT "" CACHE STRING "Use alternate sysroot.")
set(LIBCXX_GCC_TOOLCHAIN "" CACHE STRING "Use alternate GCC toolchain.")

# Feature options -------------------------------------------------------------
option(LIBCXX_ENABLE_EXCEPTIONS "Use exceptions." ON)
option(LIBCXX_ENABLE_RTTI "Use run time type information." ON)
option(LIBCXX_ENABLE_GLOBAL_FILESYSTEM_NAMESPACE "Build libc++ with support for the global filesystem namespace." ON)
option(LIBCXX_ENABLE_STDIN "Build libc++ with support for stdin/std::cin." ON)
option(LIBCXX_ENABLE_STDOUT "Build libc++ with support for stdout/std::cout." ON)
option(LIBCXX_ENABLE_THREADS "Build libc++ with support for threads." ON)
option(LIBCXX_ENABLE_THREAD_UNSAFE_C_FUNCTIONS "Build libc++ with support for thread-unsafe C functions" ON)
option(LIBCXX_ENABLE_MONOTONIC_CLOCK
  "Build libc++ with support for a monotonic clock.
   This option may only be set to OFF when LIBCXX_ENABLE_THREADS=OFF." ON)
option(LIBCXX_HAS_MUSL_LIBC "Build libc++ with support for the Musl C library" OFF)
option(LIBCXX_HAS_PTHREAD_API "Ignore auto-detection and force use of pthread API" OFF)
option(LIBCXX_HAS_WIN32_THREAD_API "Ignore auto-detection and force use of win32 thread API" OFF)
option(LIBCXX_HAS_EXTERNAL_THREAD_API
  "Build libc++ with an externalized threading API.
   This option may only be set to ON when LIBCXX_ENABLE_THREADS=ON." OFF)
option(LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY
    "Build libc++ with an externalized threading library.
     This option may only be set to ON when LIBCXX_ENABLE_THREADS=ON" OFF)

# Misc options ----------------------------------------------------------------
# FIXME: Turn -pedantic back ON. It is currently off because it warns
# about #include_next which is used everywhere.
option(LIBCXX_ENABLE_PEDANTIC "Compile with pedantic enabled." OFF)
option(LIBCXX_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(LIBCXX_DISABLE_MACRO_CONFLICT_WARNINGS "Disable #warnings about conflicting macros." OFF)

option(LIBCXX_GENERATE_COVERAGE "Enable generating code coverage." OFF)
set(LIBCXX_COVERAGE_LIBRARY "" CACHE STRING
    "The Profile-rt library used to build with code coverage")

# Don't allow a user to accidentally overwrite the system libc++ installation on Darwin.
# If the user specifies -DCMAKE_INSTALL_PREFIX=/usr the install rules for libc++
# will not be generated and a warning will be issued.
option(LIBCXX_OVERRIDE_DARWIN_INSTALL "Enable overwriting darwins libc++ installation." OFF)
mark_as_advanced(LIBCXX_OVERRIDE_DARWIN_INSTALL) # Don't show this option by default.

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT LIBCXX_OVERRIDE_DARWIN_INSTALL)
  if ("${CMAKE_INSTALL_PREFIX}" STREQUAL "/usr")
    message(WARNING "Disabling libc++ install rules because installation would "
                    "overwrite the systems installation. Configure with "
                    "-DLIBCXX_OVERRIDE_DARWIN_INSTALL=ON to suppress this behaviour.")
    mark_as_advanced(CLEAR LIBCXX_OVERRIDE_DARWIN_INSTALL) # Show the override option.
    set(LIBCXX_INSTALL_HEADERS OFF)
    set(LIBCXX_INSTALL_LIBRARY OFF)
  endif()
endif()

set(LIBCXX_CONFIGURE_IDE_DEFAULT OFF)
if (XCODE OR MSVC_IDE)
  set(LIBCXX_CONFIGURE_IDE_DEFAULT ON)
endif()
option(LIBCXX_CONFIGURE_IDE "Configure libcxx for use within an IDE"
      ${LIBCXX_CONFIGURE_IDE_DEFAULT})

#===============================================================================
# Check option configurations
#===============================================================================

if (LIBCXX_ENABLE_FILESYSTEM AND NOT LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY)
  message(FATAL_ERROR
    "LIBCXX_ENABLE_FILESYSTEM cannot be turned on when LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY=OFF")
endif()

# Ensure LIBCXX_ENABLE_MONOTONIC_CLOCK is set to ON only when
# LIBCXX_ENABLE_THREADS is on.
if(LIBCXX_ENABLE_THREADS AND NOT LIBCXX_ENABLE_MONOTONIC_CLOCK)
  message(FATAL_ERROR "LIBCXX_ENABLE_MONOTONIC_CLOCK can only be set to OFF"
                      " when LIBCXX_ENABLE_THREADS is also set to OFF.")
endif()

if(NOT LIBCXX_ENABLE_THREADS)
  if(LIBCXX_HAS_PTHREAD_API)
    message(FATAL_ERROR "LIBCXX_HAS_PTHREAD_API can only be set to ON"
                        " when LIBCXX_ENABLE_THREADS is also set to ON.")
  endif()
  if(LIBCXX_HAS_EXTERNAL_THREAD_API)
    message(FATAL_ERROR "LIBCXX_HAS_EXTERNAL_THREAD_API can only be set to ON"
                        " when LIBCXX_ENABLE_THREADS is also set to ON.")
  endif()
  if (LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY)
    message(FATAL_ERROR "LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY can only be set "
                        "to ON when LIBCXX_ENABLE_THREADS is also set to ON.")
  endif()
  if (LIBCXX_HAS_WIN32_THREAD_API)
    message(FATAL_ERROR "LIBCXX_HAS_WIN32_THREAD_API can only be set to ON"
                        " when LIBCXX_ENABLE_THREADS is also set to ON.")
  endif()

endif()

if (LIBCXX_HAS_EXTERNAL_THREAD_API)
  if (LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY)
    message(FATAL_ERROR "The options LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY and "
                        "LIBCXX_HAS_EXTERNAL_THREAD_API cannot both be ON at "
                        "the same time")
  endif()
  if (LIBCXX_HAS_PTHREAD_API)
    message(FATAL_ERROR "The options LIBCXX_HAS_EXTERNAL_THREAD_API"
                        "and LIBCXX_HAS_PTHREAD_API cannot be both"
                        "set to ON at the same time.")
  endif()
  if (LIBCXX_HAS_WIN32_THREAD_API)
    message(FATAL_ERROR "The options LIBCXX_HAS_EXTERNAL_THREAD_API"
                        "and LIBCXX_HAS_WIN32_THREAD_API cannot be both"
                        "set to ON at the same time.")
  endif()
endif()

if (LIBCXX_HAS_PTHREAD_API)
  if (LIBCXX_HAS_WIN32_THREAD_API)
    message(FATAL_ERROR "The options LIBCXX_HAS_PTHREAD_API"
                        "and LIBCXX_HAS_WIN32_THREAD_API cannot be both"
                        "set to ON at the same time.")
  endif()
endif()

# Ensure LLVM_USE_SANITIZER is not specified when LIBCXX_GENERATE_COVERAGE
# is ON.
if (LLVM_USE_SANITIZER AND LIBCXX_GENERATE_COVERAGE)
  message(FATAL_ERROR "LLVM_USE_SANITIZER cannot be used with LIBCXX_GENERATE_COVERAGE")
endif()

# Set LIBCXX_BUILD_32_BITS to (LIBCXX_BUILD_32_BITS OR LLVM_BUILD_32_BITS)
# and check that we can build with 32 bits if requested.
if (CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32)
  if (LIBCXX_BUILD_32_BITS AND NOT LLVM_BUILD_32_BITS) # Don't duplicate the output from LLVM
    message(STATUS "Building 32 bits executables and libraries.")
  endif()
elseif(LIBCXX_BUILD_32_BITS)
  message(FATAL_ERROR "LIBCXX_BUILD_32_BITS=ON is not supported on this platform.")
endif()

# Check that this option is not enabled on Apple and emit a usage warning.
if (LIBCXX_ENABLE_STATIC_ABI_LIBRARY)
  if (APPLE)
    message(FATAL_ERROR "LIBCXX_ENABLE_STATIC_ABI_LIBRARY is not supported on OS X")
  else()
    message(WARNING "LIBCXX_ENABLE_STATIC_ABI_LIBRARY is an experimental option")
  endif()
  if (LIBCXX_ENABLE_STATIC AND NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "LIBCXX_ENABLE_STATIC_ABI_LIBRARY requires python but it was not found.")
  endif()
endif()

if (LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
    if (APPLE)
      message(FATAL_ERROR "LIBCXX_ENABLE_ABI_LINKER_SCRIPT cannot be used on APPLE targets")
    endif()
    if (NOT PYTHONINTERP_FOUND)
      message(FATAL_ERROR "LIBCXX_ENABLE_ABI_LINKER_SCRIPT requires python but it was not found.")
    endif()
    if (NOT LIBCXX_ENABLE_SHARED)
      message(FATAL_ERROR "LIBCXX_ENABLE_ABI_LINKER_SCRIPT is only available for shared library builds.")
    endif()
endif()

if (LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY AND LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
    message(FATAL_ERROR "Conflicting options given.
        LIBCXX_ENABLE_STATIC_ABI_LIBRARY cannot be specified with
        LIBCXX_ENABLE_ABI_LINKER_SCRIPT")
endif()

if (LIBCXX_HAS_MUSL_LIBC AND NOT LIBCXX_INSTALL_SUPPORT_HEADERS)
  message(FATAL_ERROR "LIBCXX_INSTALL_SUPPORT_HEADERS can not be turned off"
                      "when building for Musl with LIBCXX_HAS_MUSL_LIBC.")
endif()

if (LIBCXX_ABI_FORCE_ITANIUM AND LIBCXX_ABI_FORCE_MICROSOFT)
  message(FATAL_ERROR "Only one of LIBCXX_ABI_FORCE_ITANIUM and LIBCXX_ABI_FORCE_MICROSOFT can be specified.")
endif ()

#===============================================================================
# Configure System
#===============================================================================

set(LIBCXX_COMPILER    ${CMAKE_CXX_COMPILER})
set(LIBCXX_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBCXX_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(LIBCXX_BINARY_INCLUDE_DIR "${LIBCXX_BINARY_DIR}/include/c++build")

string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" CLANG_VERSION
       ${PACKAGE_VERSION})

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(DEFAULT_INSTALL_PREFIX lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/${LLVM_DEFAULT_TARGET_TRIPLE}/)
  set(DEFAULT_INSTALL_HEADER_PREFIX lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/)
  set(LIBCXX_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/${LLVM_DEFAULT_TARGET_TRIPLE}/lib${LIBCXX_LIBDIR_SUFFIX})
  set(LIBCXX_HEADER_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION})
elseif(LLVM_LIBRARY_OUTPUT_INTDIR)
  set(LIBCXX_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  set(LIBCXX_HEADER_DIR  ${LLVM_BINARY_DIR})
else()
  set(LIBCXX_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBCXX_LIBDIR_SUFFIX})
endif()

file(MAKE_DIRECTORY "${LIBCXX_BINARY_INCLUDE_DIR}")

set(LIBCXX_INSTALL_PREFIX ${DEFAULT_INSTALL_PREFIX} CACHE STRING
    "Define libc++ destination prefix.")

set(LIBCXX_INSTALL_HEADER_PREFIX ${DEFAULT_INSTALL_HEADER_PREFIX} CACHE STRING
    "Define libc++ header destination prefix.")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBCXX_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBCXX_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBCXX_LIBRARY_DIR})

# Declare libc++ configuration variables.
# They are intended for use as follows:
# LIBCXX_CXX_FLAGS: General flags for both the compiler and linker.
# LIBCXX_COMPILE_FLAGS: Compile only flags.
# LIBCXX_LINK_FLAGS: Linker only flags.
# LIBCXX_LIBRARIES: libraries libc++ is linked to.
# LIBCXX_INTERFACE_LIBRARIES: Libraries that must be linked when using libc++
#                             These libraries are exposed in the linker script.
set(LIBCXX_COMPILE_FLAGS "")
set(LIBCXX_LINK_FLAGS "")
set(LIBCXX_LIBRARIES "")
set(LIBCXX_INTERFACE_LIBRARIES "")

# Include macros for adding and removing libc++ flags.
include(HandleLibcxxFlags)

# Target flags ================================================================
# These flags get added to CMAKE_CXX_FLAGS and CMAKE_C_FLAGS so that
# 'config-ix' use them during feature checks. It also adds them to both
# 'LIBCXX_COMPILE_FLAGS' and 'LIBCXX_LINK_FLAGS'
add_target_flags_if(LIBCXX_BUILD_32_BITS "-m32")
add_target_flags_if(LIBCXX_TARGET_TRIPLE "--target=${LIBCXX_TARGET_TRIPLE}")
add_target_flags_if(LIBCXX_SYSROOT "--sysroot=${LIBCXX_SYSROOT}")
add_target_flags_if(LIBCXX_GCC_TOOLCHAIN "--gcc-toolchain=${LIBCXX_GCC_TOOLCHAIN}")
if (LIBCXX_TARGET_TRIPLE)
  set(TARGET_TRIPLE "${LIBCXX_TARGET_TRIPLE}")
endif()

# Configure compiler.
include(config-ix)

if (LIBCXX_USE_COMPILER_RT)
  list(APPEND LIBCXX_LINK_FLAGS "-rtlib=compiler-rt")
endif()

# Configure coverage options.
if (LIBCXX_GENERATE_COVERAGE)
  include(CodeCoverage)
  set(CMAKE_BUILD_TYPE "COVERAGE" CACHE STRING "" FORCE)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(LIBCXX_DEBUG_BUILD ON)
else()
  set(LIBCXX_DEBUG_BUILD OFF)
endif()

#===============================================================================
# Setup Compiler Flags
#===============================================================================

include(HandleLibCXXABI) # Setup the ABI library flags

if (NOT LIBCXX_STANDALONE_BUILD)
  # Remove flags that may have snuck in.
  remove_flags(-DNDEBUG -UNDEBUG -D_DEBUG
               -lc++abi)
endif()
remove_flags(-stdlib=libc++ -stdlib=libstdc++)

# FIXME: Remove all debug flags and flags that change which Windows
# default libraries are linked. Currently we only support linking the
# non-debug DLLs
remove_flags("/D_DEBUG" "/MTd" "/MDd" "/MT" "/Md")

# FIXME(EricWF): See the FIXME on LIBCXX_ENABLE_PEDANTIC.
# Remove the -pedantic flag and -Wno-pedantic and -pedantic-errors
# so they don't get transformed into -Wno and -errors respectively.
remove_flags(-Wno-pedantic -pedantic-errors -pedantic)

# Required flags ==============================================================
set(LIBCXX_STANDARD_VER c++11 CACHE INTERNAL "internal option to change build dialect")
if (LIBCXX_HAS_MUSL_LIBC)
  # musl's pthread implementations uses volatile types in their structs which is
  # not a constexpr in C++11 but is in C++14, so we use C++14 with musl.
  set(LIBCXX_STANDARD_VER c++14 CACHE INTERNAL "internal option to change build dialect")
endif()
add_compile_flags_if_supported(-std=${LIBCXX_STANDARD_VER})
mangle_name("LIBCXX_SUPPORTS_STD_EQ_${LIBCXX_STANDARD_VER}_FLAG" SUPPORTS_DIALECT_NAME)
if(NOT ${SUPPORTS_DIALECT_NAME})
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" AND NOT "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
    message(FATAL_ERROR "C++11 or greater is required but the compiler does not support ${LIBCXX_STANDARD_VER}")
  endif()
endif()

# On all systems the system c++ standard library headers need to be excluded.
# MSVC only has -X, which disables all default includes; including the crt.
# Thus, we do nothing and hope we don't accidentally include any of the C++
# headers
add_compile_flags_if_supported(-nostdinc++)

# Hide all inline function definitions which have not explicitly been marked
# visible. This prevents new definitions for inline functions from appearing in
# the dylib when get ODR used by another function.
add_compile_flags_if_supported(-fvisibility-inlines-hidden)

if (LIBCXX_CONFIGURE_IDE)
  # This simply allows IDE to process <experimental/coroutine>
  add_compile_flags_if_supported(-fcoroutines-ts)
endif()

# Let the library headers know they are currently being used to build the
# library.
add_definitions(-D_LIBCPP_BUILDING_LIBRARY)

if (NOT LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS)
  add_definitions(-D_LIBCPP_DISABLE_NEW_DELETE_DEFINITIONS)
endif()

# Warning flags ===============================================================
add_definitions(-D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
add_compile_flags_if_supported(
    -Wall -Wextra -W -Wwrite-strings
    -Wno-unused-parameter -Wno-long-long
    -Werror=return-type)
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_compile_flags_if_supported(
        -Wno-user-defined-literals
        -Wno-covered-switch-default)
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    add_compile_flags_if_supported(
        -Wno-literal-suffix
        -Wno-c++14-compat
        -Wno-noexcept-type)
endif()
if (LIBCXX_ENABLE_WERROR)
  add_compile_flags_if_supported(-Werror)
  add_compile_flags_if_supported(-WX)
else()
  # TODO(EricWF) Remove this. We shouldn't be suppressing errors when -Werror is
  # added elsewhere.
  add_compile_flags_if_supported(-Wno-error)
endif()
if (LIBCXX_ENABLE_PEDANTIC)
  add_compile_flags_if_supported(-pedantic)
endif()
if (LIBCXX_DISABLE_MACRO_CONFLICT_WARNINGS)
  add_definitions(-D_LIBCPP_DISABLE_MACRO_CONFLICT_WARNINGS)
endif()

# Exception flags =============================================================
if (LIBCXX_ENABLE_EXCEPTIONS)
  # Catches C++ exceptions only and tells the compiler to assume that extern C
  # functions never throw a C++ exception.
  add_compile_flags_if_supported(-EHsc)
else()
  add_definitions(-D_LIBCPP_NO_EXCEPTIONS)
  add_compile_flags_if_supported(-EHs- -EHa-)
  add_compile_flags_if_supported(-fno-exceptions)
endif()

# RTTI flags ==================================================================
if (NOT LIBCXX_ENABLE_RTTI)
  add_definitions(-D_LIBCPP_NO_RTTI)
  add_compile_flags_if_supported(-GR-)
  add_compile_flags_if_supported(-fno-rtti)
endif()

# Threading flags =============================================================
if (LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY AND LIBCXX_ENABLE_SHARED)
  # Need to allow unresolved symbols if this is to work with shared library builds
  if (APPLE)
    add_link_flags("-undefined dynamic_lookup")
  else()
    # Relax this restriction from HandleLLVMOptions
    string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  endif()
endif()

# Assertion flags =============================================================
define_if(LIBCXX_ENABLE_ASSERTIONS -UNDEBUG)
define_if_not(LIBCXX_ENABLE_ASSERTIONS -DNDEBUG)
define_if(LIBCXX_ENABLE_ASSERTIONS -D_LIBCPP_DEBUG=0)
define_if(LIBCXX_DEBUG_BUILD -D_DEBUG)
if (LIBCXX_ENABLE_ASSERTIONS AND NOT LIBCXX_DEBUG_BUILD)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  define_if_not(LIBCXX_TARGETING_MSVC -D_DEBUG)
endif()

# Modules flags ===============================================================
# FIXME The libc++ sources are fundamentally non-modular. They need special
# versions of the headers in order to provide C++03 and legacy ABI definitions.
# NOTE: The public headers can be used with modules in all other contexts.
if (LLVM_ENABLE_MODULES)
  # Ignore that the rest of the modules flags are now unused.
  add_compile_flags_if_supported(-Wno-unused-command-line-argument)
  add_compile_flags(-fno-modules)
endif()

# Sanitizer flags =============================================================

# Configure for sanitizers. If LIBCXX_STANDALONE_BUILD then we have to do
# the flag translation ourselves. Othewise LLVM's CMakeList.txt will handle it.
if (LIBCXX_STANDALONE_BUILD)
  set(LLVM_USE_SANITIZER "" CACHE STRING
      "Define the sanitizer used to build the library and tests")
  # NOTE: LLVM_USE_SANITIZER checks for a UNIX like system instead of MSVC.
  # But we don't have LLVM_ON_UNIX so checking for MSVC is the best we can do.
  if (LLVM_USE_SANITIZER AND NOT MSVC)
    add_flags_if_supported("-fno-omit-frame-pointer")
    add_flags_if_supported("-gline-tables-only")

    if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND
        NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
        add_flags_if_supported("-gline-tables-only")
    endif()
    if (LLVM_USE_SANITIZER STREQUAL "Address")
      add_flags("-fsanitize=address")
    elseif (LLVM_USE_SANITIZER MATCHES "Memory(WithOrigins)?")
      add_flags(-fsanitize=memory)
      if (LLVM_USE_SANITIZER STREQUAL "MemoryWithOrigins")
        add_flags("-fsanitize-memory-track-origins")
      endif()
    elseif (LLVM_USE_SANITIZER STREQUAL "Undefined")
      add_flags("-fsanitize=undefined -fno-sanitize=vptr,function -fno-sanitize-recover=all")
    elseif (LLVM_USE_SANITIZER STREQUAL "Thread")
      add_flags(-fsanitize=thread)
    else()
      message(WARNING "Unsupported value of LLVM_USE_SANITIZER: ${LLVM_USE_SANITIZER}")
    endif()
  elseif(LLVM_USE_SANITIZER AND MSVC)
    message(WARNING "LLVM_USE_SANITIZER is not supported on this platform.")
  endif()
endif()

# Configuration file flags =====================================================
if (NOT LIBCXX_ABI_VERSION EQUAL DEFAULT_ABI_VERSION)
  config_define(${LIBCXX_ABI_VERSION} _LIBCPP_ABI_VERSION)
endif()
config_define_if(LIBCXX_ABI_UNSTABLE _LIBCPP_ABI_UNSTABLE)
config_define_if(LIBCXX_ABI_FORCE_ITANIUM _LIBCPP_ABI_FORCE_ITANIUM)
config_define_if(LIBCXX_ABI_FORCE_MICROSOFT _LIBCPP_ABI_FORCE_MICROSOFT)

config_define_if_not(LIBCXX_ENABLE_GLOBAL_FILESYSTEM_NAMESPACE _LIBCPP_HAS_NO_GLOBAL_FILESYSTEM_NAMESPACE)
config_define_if_not(LIBCXX_ENABLE_STDIN _LIBCPP_HAS_NO_STDIN)
config_define_if_not(LIBCXX_ENABLE_STDOUT _LIBCPP_HAS_NO_STDOUT)
config_define_if_not(LIBCXX_ENABLE_THREADS _LIBCPP_HAS_NO_THREADS)
config_define_if_not(LIBCXX_ENABLE_MONOTONIC_CLOCK _LIBCPP_HAS_NO_MONOTONIC_CLOCK)
config_define_if_not(LIBCXX_ENABLE_THREAD_UNSAFE_C_FUNCTIONS _LIBCPP_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS)

config_define_if(LIBCXX_HAS_PTHREAD_API _LIBCPP_HAS_THREAD_API_PTHREAD)
config_define_if(LIBCXX_HAS_EXTERNAL_THREAD_API _LIBCPP_HAS_THREAD_API_EXTERNAL)
config_define_if(LIBCXX_HAS_WIN32_THREAD_API _LIBCPP_HAS_THREAD_API_WIN32)
config_define_if(LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY _LIBCPP_HAS_THREAD_LIBRARY_EXTERNAL)
config_define_if(LIBCXX_HAS_MUSL_LIBC _LIBCPP_HAS_MUSL_LIBC)
config_define_if(LIBCXX_NO_VCRUNTIME _LIBCPP_NO_VCRUNTIME)

if (LIBCXX_ABI_DEFINES)
  set(abi_defines)
  foreach (abi_define ${LIBCXX_ABI_DEFINES})
    if (NOT abi_define MATCHES "^_LIBCPP_ABI_")
      message(SEND_ERROR "Invalid ABI macro ${abi_define} in LIBCXX_ABI_DEFINES")
    endif()
    list(APPEND abi_defines "#define ${abi_define}")
  endforeach()
  string(REPLACE ";" "\n" abi_defines "${abi_defines}")
  config_define(${abi_defines} _LIBCPP_ABI_DEFINES)
endif()

# By default libc++ on Windows expects to use a shared library, which requires
# the headers to use DLL import/export semantics. However when building a
# static library only we modify the headers to disable DLL import/export.
if (DEFINED WIN32 AND LIBCXX_ENABLE_STATIC AND NOT LIBCXX_ENABLE_SHARED)
  message(STATUS "Generating custom __config for non-DLL Windows build")
  config_define(ON _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS)
endif()

set(site_config_path "${LIBCXX_BINARY_DIR}/__config_site")
if (LIBCXX_NEEDS_SITE_CONFIG)
  configure_file("include/__config_site.in"
                 "${site_config_path}"
                 @ONLY)

  # Provide the config definitions by included the generated __config_site
  # file at compile time.
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC" OR "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
    add_compile_flags("/FI\"${LIBCXX_BINARY_DIR}/__config_site\"")
  else()
    add_compile_flags("-include ${LIBCXX_BINARY_DIR}/__config_site")
  endif()
else()
  if (EXISTS "${site_config_path}")
    message(STATUS "Removing stale site configuration ${site_config_path}")
    file(REMOVE "${site_config_path}")
  endif()
endif()

#===============================================================================
# Setup Source Code And Tests
#===============================================================================
include_directories(include)
add_subdirectory(include)
add_subdirectory(lib)


if (LIBCXX_INCLUDE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Create the lit.site.cfg file even when LIBCXX_INCLUDE_TESTS is OFF or
# LLVM_FOUND is OFF. This allows users to run the tests manually using
# LIT without requiring a full LLVM checkout.
#
# However, since some submission systems strip test/ subdirectories, check for
# it before adding it.

if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test")
  add_subdirectory(test)
endif()
if (LIBCXX_INCLUDE_TESTS)
  add_subdirectory(lib/abi)
endif()

if (LIBCXX_STANDALONE_BUILD AND EXISTS "${LLVM_MAIN_SRC_DIR}/utils/llvm-lit")
  include(AddLLVM) # for get_llvm_lit_path
  # Make sure the llvm-lit script is generated into the bin directory, and do
  # it after adding all tests, since the generated script will only work
  # correctly discovered tests against test locations from the source tree that
  # have already been discovered.
  add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit
                   ${CMAKE_CURRENT_BINARY_DIR}/llvm-lit)
endif()

if (LIBCXX_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()
