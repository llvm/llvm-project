# See https://libcxx.llvm.org/docs/BuildingLibcxx.html for instructions on how
# to build libcxx with CMake.

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../libcxxabi")
  message(FATAL_ERROR "libc++ now requires being built in a monorepo layout with libcxxabi available")
endif()

#===============================================================================
# Setup Project
#===============================================================================
cmake_minimum_required(VERSION 3.13.4)

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )

set(CMAKE_FOLDER "libc++")

set(LIBCXX_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBCXX_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(LIBCXX_BINARY_INCLUDE_DIR "${LIBCXX_BINARY_DIR}/include/c++build")

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR LIBCXX_STANDALONE_BUILD)
  project(libcxx CXX C)

  set(PACKAGE_NAME libcxx)
  set(PACKAGE_VERSION 14.0.0git)
  set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
  set(PACKAGE_BUGREPORT "llvm-bugs@lists.llvm.org")

  # In a standalone build, we don't have llvm to automatically generate the
  # llvm-lit script for us.  So we need to provide an explicit directory that
  # the configurator should write the script into.
  set(LIBCXX_STANDALONE_BUILD 1)
  set(LLVM_LIT_OUTPUT_DIR "${LIBCXX_BINARY_DIR}/bin")

  # Find the LLVM sources and simulate LLVM CMake options.
  include(HandleOutOfTreeLLVM)
endif()

if (LIBCXX_STANDALONE_BUILD)
  find_package(Python3 COMPONENTS Interpreter)
  if(NOT Python3_Interpreter_FOUND)
    message(WARNING "Python3 not found, using python2 as a fallback")
    find_package(Python2 COMPONENTS Interpreter REQUIRED)
    if(Python2_VERSION VERSION_LESS 2.7)
      message(SEND_ERROR "Python 2.7 or newer is required")
    endif()

    # Treat python2 as python3
    add_executable(Python3::Interpreter IMPORTED)
    set_target_properties(Python3::Interpreter PROPERTIES
      IMPORTED_LOCATION ${Python2_EXECUTABLE})
    set(Python3_EXECUTABLE ${Python2_EXECUTABLE})
  endif()
endif()

# Require out of source build.
include(MacroEnsureOutOfSourceBuild)
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate
 build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND "${CMAKE_CXX_SIMULATE_ID}" STREQUAL "MSVC")
  message(STATUS "Configuring for clang-cl")
  set(LIBCXX_TARGETING_CLANG_CL ON)
endif()

if (MSVC)
  set(LIBCXX_TARGETING_MSVC ON)
  message(STATUS "Configuring for MSVC")
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
set(ENABLE_FILESYSTEM_DEFAULT ON)
if (WIN32 AND NOT MINGW)
  # Filesystem is buildable for windows, but it requires __int128 helper
  # functions, that currently are provided by libgcc or compiler_rt builtins.
  # These are available in MinGW environments, but not currently in MSVC
  # environments.
  set(ENABLE_FILESYSTEM_DEFAULT OFF)
endif()
option(LIBCXX_ENABLE_FILESYSTEM "Build filesystem as part of the main libc++ library"
    ${ENABLE_FILESYSTEM_DEFAULT})
option(LIBCXX_INCLUDE_TESTS "Build the libc++ tests." ${LLVM_INCLUDE_TESTS})
option(LIBCXX_ENABLE_PARALLEL_ALGORITHMS "Enable the parallel algorithms library. This requires the PSTL to be available." OFF)
option(LIBCXX_ENABLE_DEBUG_MODE_SUPPORT
  "Whether to include support for libc++'s debugging mode in the library.
   By default, this is turned on. If you turn it off and try to enable the
   debug mode when compiling a program against libc++, it will fail to link
   since the required support isn't provided in the library." ON)
option(LIBCXX_ENABLE_RANDOM_DEVICE
  "Whether to include support for std::random_device in the library. Disabling
   this can be useful when building the library for platforms that don't have
   a source of randomness, such as some embedded platforms. When this is not
   supported, most of <random> will still be available, but std::random_device
   will not." ON)
option(LIBCXX_ENABLE_LOCALIZATION
  "Whether to include support for localization in the library. Disabling
   localization can be useful when porting to platforms that don't support
   the C locale API (e.g. embedded). When localization is not supported,
   several parts of the library will be disabled: <iostream>, <regex>, <locale>
   will be completely unusable, and other parts may be only partly available." ON)
option(LIBCXX_ENABLE_UNICODE
  "Whether to include support for Unicode in the library. Disabling Unicode can
   be useful when porting to platforms that don't support UTF-8 encoding (e.g.
   embedded)." ON)
option(LIBCXX_ENABLE_VENDOR_AVAILABILITY_ANNOTATIONS
  "Whether to turn on vendor availability annotations on declarations that depend
   on definitions in a shared library. By default, we assume that we're not building
   libc++ for any specific vendor, and we disable those annotations. Vendors wishing
   to provide compile-time errors when using features unavailable on some version of
   the shared library they shipped should turn this on and see `include/__availability`
   for more details." OFF)
option(LIBCXX_ENABLE_INCOMPLETE_FEATURES
    "Whether to enable support for incomplete library features. Incomplete features
    are new library features under development. These features don't guarantee
    ABI stability nor the quality of completed library features. Vendors
    shipping the library may want to disable this option." ON)
set(LIBCXX_TEST_CONFIG "legacy.cfg.in" CACHE STRING
    "The path to the Lit testing configuration to use when running the tests.
     If a relative path is provided, it is assumed to be relative to '<monorepo>/libcxx/test/configs'.")
if (NOT IS_ABSOLUTE "${LIBCXX_TEST_CONFIG}")
  set(LIBCXX_TEST_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/test/configs/${LIBCXX_TEST_CONFIG}")
endif()
set(LIBCXX_TEST_PARAMS "" CACHE STRING
    "A list of parameters to run the Lit test suite with.")

# Benchmark options -----------------------------------------------------------
option(LIBCXX_INCLUDE_BENCHMARKS "Build the libc++ benchmarks and their dependencies" ON)

set(LIBCXX_BENCHMARK_TEST_ARGS_DEFAULT --benchmark_min_time=0.01)
set(LIBCXX_BENCHMARK_TEST_ARGS "${LIBCXX_BENCHMARK_TEST_ARGS_DEFAULT}" CACHE STRING
    "Arguments to pass when running the benchmarks using check-cxx-benchmarks")

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
cmake_dependent_option(LIBCXX_INSTALL_EXPERIMENTAL_LIBRARY
        "Install libc++experimental.a" ON
        "LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY;LIBCXX_INSTALL_LIBRARY" OFF)

set(LIBCXX_ABI_VERSION "1" CACHE STRING "ABI version of libc++. Can be either 1 or 2, where 2 is currently not stable. Defaults to 1.")
set(LIBCXX_ABI_NAMESPACE "" CACHE STRING "The inline ABI namespace used by libc++. It defaults to __n where `n` is the current ABI version.")
option(LIBCXX_ABI_UNSTABLE "Unstable ABI of libc++." OFF)
option(LIBCXX_ABI_FORCE_ITANIUM "Ignore auto-detection and force use of the Itanium ABI.")
option(LIBCXX_ABI_FORCE_MICROSOFT "Ignore auto-detection and force use of the Microsoft ABI.")

set(LIBCXX_TYPEINFO_COMPARISON_IMPLEMENTATION "default" CACHE STRING
  "Override the implementation to use for comparing typeinfos. By default, this
   is detected automatically by the library, but this option allows overriding
   which implementation is used unconditionally.

   See the documentation in <libcxx/include/typeinfo> for details on what each
   value means.")
set(TYPEINFO_COMPARISON_VALUES "default;1;2;3")
if (NOT ("${LIBCXX_TYPEINFO_COMPARISON_IMPLEMENTATION}" IN_LIST TYPEINFO_COMPARISON_VALUES))
  message(FATAL_ERROR "Value '${LIBCXX_TYPEINFO_COMPARISON_IMPLEMENTATION}' is not a valid value for
                       LIBCXX_TYPEINFO_COMPARISON_IMPLEMENTATION")
endif()

option(LIBCXX_HIDE_FROM_ABI_PER_TU_BY_DEFAULT "Enable per TU ABI insulation by default. To be used by vendors." OFF)
set(LIBCXX_ABI_DEFINES "" CACHE STRING "A semicolon separated list of ABI macros to define in the site config header.")
option(LIBCXX_USE_COMPILER_RT "Use compiler-rt instead of libgcc" OFF)
set(LIBCXX_LIBCPPABI_VERSION "2" CACHE STRING "Version of libc++abi's ABI to re-export from libc++ when re-exporting is enabled.
                                               Note that this is not related to the version of libc++'s ABI itself!")

# ABI Library options ---------------------------------------------------------
set(LIBCXX_CXX_ABI "default" CACHE STRING "Specify C++ ABI library to use.")
set(CXXABIS none default libcxxabi libcxxrt libstdc++ libsupc++ vcruntime)
set_property(CACHE LIBCXX_CXX_ABI PROPERTY STRINGS ;${CXXABIS})

# Setup the default options if LIBCXX_CXX_ABI is not specified.
if (LIBCXX_CXX_ABI STREQUAL "default")
  if (LIBCXX_TARGETING_MSVC)
    # FIXME: Figure out how to configure the ABI library on Windows.
    set(LIBCXX_CXX_ABI_LIBNAME "vcruntime")
  elseif (${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
    set(LIBCXX_CXX_ABI_LIBNAME "libcxxrt")
  elseif (NOT LIBCXX_STANDALONE_BUILD OR HAVE_LIBCXXABI)
    set(LIBCXX_CXX_ABI_LIBNAME "libcxxabi")
  else()
    set(LIBCXX_CXX_ABI_LIBNAME "default")
  endif()
else()
  set(LIBCXX_CXX_ABI_LIBNAME "${LIBCXX_CXX_ABI}")
endif()

option(LIBCXX_ENABLE_STATIC_ABI_LIBRARY
  "Use a static copy of the ABI library when linking libc++.
   This option cannot be used with LIBCXX_ENABLE_ABI_LINKER_SCRIPT." OFF)

cmake_dependent_option(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY
  "Statically link the ABI library to static library" ON
  "LIBCXX_ENABLE_STATIC_ABI_LIBRARY;LIBCXX_ENABLE_STATIC" OFF)

cmake_dependent_option(LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY
  "Statically link the ABI library to shared library" ON
  "LIBCXX_ENABLE_STATIC_ABI_LIBRARY;LIBCXX_ENABLE_SHARED" OFF)

# Generate and install a linker script inplace of libc++.so. The linker script
# will link libc++ to the correct ABI library. This option is on by default
# on UNIX platforms other than Apple unless 'LIBCXX_ENABLE_STATIC_ABI_LIBRARY'
# is on. This option is also disabled when the ABI library is not specified
# or is specified to be "none".
set(ENABLE_LINKER_SCRIPT_DEFAULT_VALUE OFF)
if (LLVM_HAVE_LINK_VERSION_SCRIPT AND NOT LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY
      AND NOT LIBCXX_CXX_ABI_LIBNAME STREQUAL "none"
      AND NOT LIBCXX_CXX_ABI_LIBNAME STREQUAL "default"
      AND Python3_EXECUTABLE
      AND LIBCXX_ENABLE_SHARED)
    set(ENABLE_LINKER_SCRIPT_DEFAULT_VALUE ON)
endif()

option(LIBCXX_ENABLE_ABI_LINKER_SCRIPT
      "Use and install a linker script for the given ABI library"
      ${ENABLE_LINKER_SCRIPT_DEFAULT_VALUE})

option(LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS
  "Build libc++ with definitions for operator new/delete. These are normally
   defined in libc++abi, but this option can be used to define them in libc++
   instead. If you define them in libc++, make sure they are NOT defined in
   libc++abi. Doing otherwise is an ODR violation." OFF)
# Build libc++abi with libunwind. We need this option to determine whether to
# link with libunwind or libgcc_s while running the test cases.
option(LIBCXXABI_USE_LLVM_UNWINDER "Build and use the LLVM unwinder." OFF)

# Target options --------------------------------------------------------------
option(LIBCXX_BUILD_32_BITS "Build 32 bit libc++." ${LLVM_BUILD_32_BITS})
set(LIBCXX_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}" CACHE STRING "Use alternate target triple.")
set(LIBCXX_SYSROOT "" CACHE STRING "Use alternate sysroot.")
set(LIBCXX_GCC_TOOLCHAIN "" CACHE STRING "Use alternate GCC toolchain.")

# Feature options -------------------------------------------------------------
option(LIBCXX_ENABLE_EXCEPTIONS "Use exceptions." ON)
option(LIBCXX_ENABLE_RTTI "Use run time type information." ON)
option(LIBCXX_ENABLE_THREADS "Build libc++ with support for threads." ON)
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

set(LIBCXX_CONFIGURE_IDE_DEFAULT OFF)
if (XCODE OR MSVC_IDE)
  set(LIBCXX_CONFIGURE_IDE_DEFAULT ON)
endif()
option(LIBCXX_CONFIGURE_IDE "Configure libcxx for use within an IDE"
      ${LIBCXX_CONFIGURE_IDE_DEFAULT})

option(LIBCXX_HERMETIC_STATIC_LIBRARY
  "Do not export any symbols from the static library." OFF)

#===============================================================================
# Check option configurations
#===============================================================================

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

# Warn users that LIBCXX_ENABLE_STATIC_ABI_LIBRARY is an experimental option.
if (LIBCXX_ENABLE_STATIC_ABI_LIBRARY)
  message(WARNING "LIBCXX_ENABLE_STATIC_ABI_LIBRARY is an experimental option")
  if (LIBCXX_ENABLE_STATIC AND NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "LIBCXX_ENABLE_STATIC_ABI_LIBRARY requires python but it was not found.")
  endif()
endif()

if (LIBCXX_ENABLE_ABI_LINKER_SCRIPT)
    if (APPLE)
      message(FATAL_ERROR "LIBCXX_ENABLE_ABI_LINKER_SCRIPT cannot be used on APPLE targets")
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

if (LIBCXX_ABI_FORCE_ITANIUM AND LIBCXX_ABI_FORCE_MICROSOFT)
  message(FATAL_ERROR "Only one of LIBCXX_ABI_FORCE_ITANIUM and LIBCXX_ABI_FORCE_MICROSOFT can be specified.")
endif ()

#===============================================================================
# Configure System
#===============================================================================

# TODO: Projects that depend on libc++ should use LIBCXX_GENERATED_INCLUDE_DIR
# instead of hard-coding include/c++/v1.
if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(LIBCXX_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE})
  set(LIBCXX_GENERATED_INCLUDE_DIR "${LLVM_BINARY_DIR}/include/c++/v1")
  set(LIBCXX_GENERATED_INCLUDE_TARGET_DIR "${LLVM_BINARY_DIR}/include/${LLVM_DEFAULT_TARGET_TRIPLE}/c++/v1")
  set(LIBCXX_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE} CACHE PATH
      "Path where built libc++ libraries should be installed.")
  set(LIBCXX_INSTALL_RUNTIME_DIR bin CACHE PATH
      "Path where built libc++ runtime libraries should be installed.")
  set(LIBCXX_INSTALL_INCLUDE_DIR "include/c++/v1" CACHE PATH
      "Path where target-agnostic libc++ headers should be installed.")
  set(LIBCXX_INSTALL_INCLUDE_TARGET_DIR "include/${LLVM_DEFAULT_TARGET_TRIPLE}/c++/v1" CACHE PATH
      "Path where target-specific libc++ headers should be installed.")
  if(LIBCXX_LIBDIR_SUBDIR)
    string(APPEND LIBCXX_LIBRARY_DIR /${LIBCXX_LIBDIR_SUBDIR})
    string(APPEND LIBCXX_INSTALL_LIBRARY_DIR /${LIBCXX_LIBDIR_SUBDIR})
  endif()
elseif(LLVM_LIBRARY_OUTPUT_INTDIR)
  set(LIBCXX_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  set(LIBCXX_GENERATED_INCLUDE_DIR "${LLVM_BINARY_DIR}/include/c++/v1")
  set(LIBCXX_GENERATED_INCLUDE_TARGET_DIR "${LIBCXX_GENERATED_INCLUDE_DIR}")
  set(LIBCXX_INSTALL_LIBRARY_DIR lib${LIBCXX_LIBDIR_SUFFIX} CACHE PATH
      "Path where built libc++ libraries should be installed.")
  set(LIBCXX_INSTALL_RUNTIME_DIR bin CACHE PATH
      "Path where built libc++ runtime libraries should be installed.")
  set(LIBCXX_INSTALL_INCLUDE_DIR "include/c++/v1" CACHE PATH
      "Path where target-agnostic libc++ headers should be installed.")
  set(LIBCXX_INSTALL_INCLUDE_TARGET_DIR "${LIBCXX_INSTALL_INCLUDE_DIR}" CACHE PATH
      "Path where target-specific libc++ headers should be installed.")
else()
  set(LIBCXX_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBCXX_LIBDIR_SUFFIX})
  set(LIBCXX_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include/c++/v1")
  set(LIBCXX_GENERATED_INCLUDE_TARGET_DIR "${LIBCXX_GENERATED_INCLUDE_DIR}")
  set(LIBCXX_INSTALL_LIBRARY_DIR lib${LIBCXX_LIBDIR_SUFFIX} CACHE PATH
      "Path where built libc++ libraries should be installed.")
  set(LIBCXX_INSTALL_RUNTIME_DIR bin CACHE PATH
      "Path where built libc++ runtime libraries should be installed.")
  set(LIBCXX_INSTALL_INCLUDE_DIR "include/c++/v1" CACHE PATH
      "Path where target-agnostic libc++ headers should be installed.")
  set(LIBCXX_INSTALL_INCLUDE_TARGET_DIR "${LIBCXX_INSTALL_INCLUDE_DIR}" CACHE PATH
      "Path where target-specific libc++ headers should be installed.")
endif()

file(MAKE_DIRECTORY "${LIBCXX_BINARY_INCLUDE_DIR}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBCXX_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBCXX_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBCXX_LIBRARY_DIR})

# Declare libc++ configuration variables.
# They are intended for use as follows:
# LIBCXX_CXX_FLAGS: General flags for both the compiler and linker.
# LIBCXX_COMPILE_FLAGS: Compile only flags.
# LIBCXX_LINK_FLAGS: Linker only flags.
# LIBCXX_LIBRARIES: libraries libc++ is linked to.
set(LIBCXX_COMPILE_FLAGS "")
set(LIBCXX_LINK_FLAGS "")
set(LIBCXX_LIBRARIES "")

# Include macros for adding and removing libc++ flags.
include(HandleLibcxxFlags)

# Target flags ================================================================
# These flags get added to CMAKE_CXX_FLAGS and CMAKE_C_FLAGS so that
# 'config-ix' use them during feature checks. It also adds them to both
# 'LIBCXX_COMPILE_FLAGS' and 'LIBCXX_LINK_FLAGS'
add_target_flags_if(LIBCXX_BUILD_32_BITS "-m32")

if(LIBCXX_TARGET_TRIPLE)
  add_target_flags_if_supported("--target=${LIBCXX_TARGET_TRIPLE}")
elseif(CMAKE_CXX_COMPILER_TARGET)
  set(LIBCXX_TARGET_TRIPLE "${CMAKE_CXX_COMPILER_TARGET}")
endif()
if(LIBCXX_SYSROOT)
  add_target_flags_if_supported("--sysroot=${LIBCXX_SYSROOT}")
elseif(CMAKE_SYSROOT)
  set(LIBCXX_SYSROOT "${CMAKE_SYSROOT}")
endif()
if(LIBCXX_GCC_TOOLCHAIN)
  add_target_flags_if_supported("--gcc-toolchain=${LIBCXX_GCC_TOOLCHAIN}")
elseif(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
  set(LIBCXX_GCC_TOOLCHAIN "${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}")
endif()

# Configure compiler.
include(config-ix)

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
remove_flags(--stdlib=libc++ -stdlib=libc++ --stdlib=libstdc++ -stdlib=libstdc++)

# FIXME: Remove all debug flags and flags that change which Windows
# default libraries are linked. Currently we only support linking the
# non-debug DLLs
remove_flags("/D_DEBUG" "/MTd" "/MDd" "/MT" "/Md")

# FIXME(EricWF): See the FIXME on LIBCXX_ENABLE_PEDANTIC.
# Remove the -pedantic flag and -Wno-pedantic and -pedantic-errors
# so they don't get transformed into -Wno and -errors respectively.
remove_flags(-Wno-pedantic -pedantic-errors -pedantic)

# Required flags ==============================================================
function(cxx_add_basic_build_flags target)

  # Require C++20 for all targets. C++17 is needed to use aligned allocation
  # in the dylib. C++20 is needed to use char8_t.
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)

  # When building the dylib, don't warn for unavailable aligned allocation
  # functions based on the deployment target -- they are always available
  # because they are provided by the dylib itself with the excepton of z/OS.
  if (ZOS)
    target_add_compile_flags_if_supported(${target} PRIVATE -fno-aligned-allocation)
  else()
    target_add_compile_flags_if_supported(${target} PRIVATE -faligned-allocation)
  endif()

  # On all systems the system c++ standard library headers need to be excluded.
  # MSVC only has -X, which disables all default includes; including the crt.
  # Thus, we do nothing and hope we don't accidentally include any of the C++
  # headers
  target_add_compile_flags_if_supported(${target} PUBLIC -nostdinc++)

  # Hide all inline function definitions which have not explicitly been marked
  # visible. This prevents new definitions for inline functions from appearing in
  # the dylib when get ODR used by another function.
  target_add_compile_flags_if_supported(${target} PRIVATE -fvisibility-inlines-hidden)

  # Our visibility annotations are not quite right for non-Clang compilers,
  # so we end up not exporting all the symbols we should. In the future, we
  # can improve the situation by providing an explicit list of exported
  # symbols on all compilers.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_add_compile_flags_if_supported(${target} PRIVATE -fvisibility=hidden)
  endif()

  if (LIBCXX_CONFIGURE_IDE)
    # This simply allows IDE to process <experimental/coroutine>
    target_add_compile_flags_if_supported(${target} PRIVATE -fcoroutines-ts)
  endif()

  # Let the library headers know they are currently being used to build the
  # library.
  target_compile_definitions(${target} PRIVATE -D_LIBCPP_BUILDING_LIBRARY)

  if (NOT LIBCXX_ENABLE_NEW_DELETE_DEFINITIONS)
    target_compile_definitions(${target} PRIVATE -D_LIBCPP_DISABLE_NEW_DELETE_DEFINITIONS)
  endif()

  if (LIBCXX_HAS_COMMENT_LIB_PRAGMA)
    if (LIBCXX_HAS_PTHREAD_LIB)
      target_compile_definitions(${target} PRIVATE -D_LIBCPP_LINK_PTHREAD_LIB)
    endif()
    if (LIBCXX_HAS_RT_LIB)
      target_compile_definitions(${target} PRIVATE -D_LIBCPP_LINK_RT_LIB)
    endif()
  endif()
endfunction()

# Warning flags ===============================================================
function(cxx_add_warning_flags target)
  target_compile_definitions(${target} PUBLIC -D_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
  if (MSVC)
    # -W4 is the cl.exe/clang-cl equivalent of -Wall. (In cl.exe and clang-cl,
    # -Wall is equivalent to -Weverything in GCC style compiler drivers.)
    target_add_compile_flags_if_supported(${target} PRIVATE -W4)
  else()
    target_add_compile_flags_if_supported(${target} PRIVATE -Wall)
  endif()
  target_add_compile_flags_if_supported(${target} PRIVATE -Wextra -W -Wwrite-strings
                                                          -Wno-unused-parameter -Wno-long-long
                                                          -Werror=return-type -Wextra-semi -Wundef)
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    target_add_compile_flags_if_supported(${target} PRIVATE
      -Wno-user-defined-literals
      -Wno-covered-switch-default
      -Wno-suggest-override
    )
    if (LIBCXX_TARGETING_CLANG_CL)
      target_add_compile_flags_if_supported(${target} PRIVATE
        -Wno-c++98-compat
        -Wno-c++98-compat-pedantic
        -Wno-c++11-compat
        -Wno-undef
        -Wno-reserved-id-macro
        -Wno-gnu-include-next
        -Wno-gcc-compat # For ignoring "'diagnose_if' is a clang extension" warnings
        -Wno-zero-as-null-pointer-constant # FIXME: Remove this and fix all occurrences.
        -Wno-deprecated-dynamic-exception-spec # For auto_ptr
        -Wno-sign-conversion
        -Wno-old-style-cast
        -Wno-deprecated # FIXME: Remove this and fix all occurrences.
        -Wno-shift-sign-overflow # FIXME: Why do we need this with clang-cl but not clang?
        -Wno-double-promotion # FIXME: remove me
      )
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    target_add_compile_flags_if_supported(${target} PRIVATE
      -Wno-literal-suffix
      -Wno-c++14-compat
      -Wno-noexcept-type
      -Wno-suggest-override)
  endif()
  if (LIBCXX_ENABLE_WERROR)
    target_add_compile_flags_if_supported(${target} PRIVATE -Werror)
    target_add_compile_flags_if_supported(${target} PRIVATE -WX)
  else()
    # TODO(EricWF) Remove this. We shouldn't be suppressing errors when -Werror is
    # added elsewhere.
    target_add_compile_flags_if_supported(${target} PRIVATE -Wno-error)
  endif()
  if (LIBCXX_ENABLE_PEDANTIC)
    target_add_compile_flags_if_supported(${target} PRIVATE -pedantic)
  endif()
  if (LIBCXX_DISABLE_MACRO_CONFLICT_WARNINGS)
    target_compile_definitions(${target} PRIVATE -D_LIBCPP_DISABLE_MACRO_CONFLICT_WARNINGS)
  endif()
endfunction()

# Exception flags =============================================================
function(cxx_add_exception_flags target)
  if (LIBCXX_ENABLE_EXCEPTIONS)
    # Catches C++ exceptions only and tells the compiler to assume that extern C
    # functions never throw a C++ exception.
    target_add_compile_flags_if_supported(${target} PUBLIC -EHsc)
  else()
    target_add_compile_flags_if_supported(${target} PUBLIC -EHs- -EHa-)
    target_add_compile_flags_if_supported(${target} PUBLIC -fno-exceptions)
  endif()
endfunction()

# RTTI flags ==================================================================
function(cxx_add_rtti_flags target)
  if (NOT LIBCXX_ENABLE_RTTI)
    target_add_compile_flags_if_supported(${target} PUBLIC -GR-)
    target_add_compile_flags_if_supported(${target} PUBLIC -fno-rtti)
  endif()
endfunction()

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
function(cxx_add_module_flags target)
  if (LLVM_ENABLE_MODULES)
    # Ignore that the rest of the modules flags are now unused.
    target_add_compile_flags_if_supported(${target} PUBLIC -Wno-unused-command-line-argument)
    target_compile_options(${target} PUBLIC -fno-modules)
  endif()
endfunction()

# Sanitizer flags =============================================================

function(get_sanitizer_flags OUT_VAR  USE_SANITIZER)
  set(SANITIZER_FLAGS)
  set(USE_SANITIZER "${USE_SANITIZER}")
  # NOTE: LLVM_USE_SANITIZER checks for a UNIX like system instead of MSVC.
  # But we don't have LLVM_ON_UNIX so checking for MSVC is the best we can do.
  if (USE_SANITIZER AND NOT MSVC)
    append_flags_if_supported(SANITIZER_FLAGS "-fno-omit-frame-pointer")
    append_flags_if_supported(SANITIZER_FLAGS "-gline-tables-only")

    if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND
            NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
      append_flags_if_supported(SANITIZER_FLAGS "-gline-tables-only")
    endif()
    if (USE_SANITIZER STREQUAL "Address")
      append_flags(SANITIZER_FLAGS "-fsanitize=address")
    elseif (USE_SANITIZER MATCHES "Memory(WithOrigins)?")
      append_flags(SANITIZER_FLAGS -fsanitize=memory)
      if (USE_SANITIZER STREQUAL "MemoryWithOrigins")
        append_flags(SANITIZER_FLAGS "-fsanitize-memory-track-origins")
      endif()
    elseif (USE_SANITIZER STREQUAL "Undefined")
      append_flags(SANITIZER_FLAGS "-fsanitize=undefined -fno-sanitize=vptr,function -fno-sanitize-recover=all")
    elseif (USE_SANITIZER STREQUAL "Address;Undefined" OR
            USE_SANITIZER STREQUAL "Undefined;Address")
      append_flags(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-sanitize=vptr,function -fno-sanitize-recover=all")
    elseif (USE_SANITIZER STREQUAL "Thread")
      append_flags(SANITIZER_FLAGS -fsanitize=thread)
    elseif (USE_SANITIZER STREQUAL "DataFlow")
      append_flags(SANITIZER_FLAGS -fsanitize=dataflow)
    else()
      message(WARNING "Unsupported value of LLVM_USE_SANITIZER: ${USE_SANITIZER}")
    endif()
  elseif(USE_SANITIZER AND MSVC)
    message(WARNING "LLVM_USE_SANITIZER is not supported on this platform.")
  endif()
  set(${OUT_VAR} "${SANITIZER_FLAGS}" PARENT_SCOPE)
endfunction()

# Configure for sanitizers. If LIBCXX_STANDALONE_BUILD then we have to do
# the flag translation ourselves. Othewise LLVM's CMakeList.txt will handle it.
if (LIBCXX_STANDALONE_BUILD)
  set(LLVM_USE_SANITIZER "" CACHE STRING
      "Define the sanitizer used to build the library and tests")
endif()
get_sanitizer_flags(SANITIZER_FLAGS "${LLVM_USE_SANITIZER}")
if (LIBCXX_STANDALONE_BUILD AND SANITIZER_FLAGS)
  add_flags(${SANITIZER_FLAGS})
endif()

# Link system libraries =======================================================
function(cxx_link_system_libraries target)

# In order to remove just libc++ from the link step
# we need to use -nostdlib++ whenever it is supported.
# Unfortunately this cannot be used universally because for example g++ supports
# only -nodefaultlibs in which case all libraries will be removed and
# all libraries but c++ have to be added in manually.
  if (LIBCXX_SUPPORTS_NOSTDLIBXX_FLAG)
    target_add_link_flags_if_supported(${target} PRIVATE "-nostdlib++")
  else()
    target_add_link_flags_if_supported(${target} PRIVATE "-nodefaultlibs")
    target_add_compile_flags_if_supported(${target} PRIVATE "/Zl")
    target_add_link_flags_if_supported(${target} PRIVATE "/nodefaultlib")
  endif()

  if (LIBCXX_HAS_SYSTEM_LIB)
    target_link_libraries(${target} PRIVATE System)
  endif()

  if (LIBCXX_HAS_PTHREAD_LIB)
    target_link_libraries(${target} PRIVATE pthread)
  endif()

  if (LIBCXX_HAS_C_LIB)
    target_link_libraries(${target} PRIVATE c)
  endif()

  if (LIBCXX_HAS_M_LIB)
    target_link_libraries(${target} PRIVATE m)
  endif()

  if (LIBCXX_HAS_RT_LIB)
    target_link_libraries(${target} PRIVATE rt)
  endif()

  if (LIBCXX_USE_COMPILER_RT)
    find_compiler_rt_library(builtins LIBCXX_BUILTINS_LIBRARY)
    if (LIBCXX_BUILTINS_LIBRARY)
      target_link_libraries(${target} PRIVATE "${LIBCXX_BUILTINS_LIBRARY}")
    endif()
  elseif (LIBCXX_HAS_GCC_LIB)
    target_link_libraries(${target} PRIVATE gcc)
  elseif (LIBCXX_HAS_GCC_S_LIB)
    target_link_libraries(${target} PRIVATE gcc_s)
  endif()

  if (LIBCXX_HAS_ATOMIC_LIB)
    target_link_libraries(${target} PRIVATE atomic)
  endif()

  if (MINGW)
    target_link_libraries(${target} PRIVATE "${MINGW_LIBRARIES}")
  endif()

  if (LIBCXX_TARGETING_MSVC)
    if (LIBCXX_DEBUG_BUILD)
      set(LIB_SUFFIX "d")
    else()
      set(LIB_SUFFIX "")
    endif()

    target_link_libraries(${target} PRIVATE ucrt${LIB_SUFFIX}) # Universal C runtime
    target_link_libraries(${target} PRIVATE vcruntime${LIB_SUFFIX}) # C++ runtime
    target_link_libraries(${target} PRIVATE msvcrt${LIB_SUFFIX}) # C runtime startup files
    target_link_libraries(${target} PRIVATE msvcprt${LIB_SUFFIX}) # C++ standard library. Required for exception_ptr internals.
    # Required for standards-complaint wide character formatting functions
    # (e.g. `printfw`/`scanfw`)
    target_link_libraries(${target} PRIVATE iso_stdio_wide_specifiers)
  endif()

  if (ANDROID AND ANDROID_PLATFORM_LEVEL LESS 21)
    target_link_libraries(${target} PUBLIC android_support)
  endif()
endfunction()

# Windows-related flags =======================================================
function(cxx_add_windows_flags target)
  if(WIN32 AND NOT MINGW)
    target_compile_definitions(${target} PRIVATE
                                 # Ignore the -MSC_VER mismatch, as we may build
                                 # with a different compatibility version.
                                 _ALLOW_MSC_VER_MISMATCH
                                 # Don't check the msvcprt iterator debug levels
                                 # as we will define the iterator types; libc++
                                 # uses a different macro to identify the debug
                                 # level.
                                 _ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH
                                 # We are building the c++ runtime, don't pull in
                                 # msvcprt.
                                 _CRTBLD
                                 # Don't warn on the use of "deprecated"
                                 # "insecure" functions which are standards
                                 # specified.
                                 _CRT_SECURE_NO_WARNINGS
                                 # Use the ISO conforming behaviour for conversion
                                 # in printf, scanf.
                                 _CRT_STDIO_ISO_WIDE_SPECIFIERS)
  endif()
endfunction()

# Configuration file flags =====================================================
if (NOT LIBCXX_ABI_VERSION EQUAL 1)
  config_define(${LIBCXX_ABI_VERSION} _LIBCPP_ABI_VERSION)
endif()
if (NOT LIBCXX_ABI_NAMESPACE STREQUAL "")
  if (NOT LIBCXX_ABI_NAMESPACE MATCHES "__.*")
    message(FATAL_ERROR "LIBCXX_ABI_NAMESPACE must be a reserved identifier.")
  endif()
  if (LIBCXX_ABI_NAMESPACE MATCHES "__[0-9]+$")
    message(FATAL_ERROR "LIBCXX_ABI_NAMESPACE '${LIBCXX_ABI_NAMESPACE}' is reserved for use by libc++.")
  endif()
  config_define(${LIBCXX_ABI_NAMESPACE} _LIBCPP_ABI_NAMESPACE)
endif()
config_define_if(LIBCXX_ABI_UNSTABLE _LIBCPP_ABI_UNSTABLE)
config_define_if(LIBCXX_ABI_FORCE_ITANIUM _LIBCPP_ABI_FORCE_ITANIUM)
config_define_if(LIBCXX_ABI_FORCE_MICROSOFT _LIBCPP_ABI_FORCE_MICROSOFT)
config_define_if(LIBCXX_HIDE_FROM_ABI_PER_TU_BY_DEFAULT _LIBCPP_HIDE_FROM_ABI_PER_TU_BY_DEFAULT)
config_define_if_not(LIBCXX_ENABLE_THREADS _LIBCPP_HAS_NO_THREADS)
config_define_if_not(LIBCXX_ENABLE_MONOTONIC_CLOCK _LIBCPP_HAS_NO_MONOTONIC_CLOCK)
if (NOT LIBCXX_TYPEINFO_COMPARISON_IMPLEMENTATION STREQUAL "default")
  config_define("${LIBCXX_TYPEINFO_COMPARISON_IMPLEMENTATION}" _LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION)
endif()
config_define_if(LIBCXX_HAS_PTHREAD_API _LIBCPP_HAS_THREAD_API_PTHREAD)
config_define_if(LIBCXX_HAS_EXTERNAL_THREAD_API _LIBCPP_HAS_THREAD_API_EXTERNAL)
config_define_if(LIBCXX_HAS_WIN32_THREAD_API _LIBCPP_HAS_THREAD_API_WIN32)
config_define_if(LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY _LIBCPP_HAS_THREAD_LIBRARY_EXTERNAL)
config_define_if(LIBCXX_HAS_MUSL_LIBC _LIBCPP_HAS_MUSL_LIBC)
config_define_if(LIBCXX_NO_VCRUNTIME _LIBCPP_NO_VCRUNTIME)
config_define_if(LIBCXX_ENABLE_PARALLEL_ALGORITHMS _LIBCPP_HAS_PARALLEL_ALGORITHMS)
config_define_if_not(LIBCXX_ENABLE_FILESYSTEM _LIBCPP_HAS_NO_FILESYSTEM_LIBRARY)
config_define_if_not(LIBCXX_ENABLE_RANDOM_DEVICE _LIBCPP_HAS_NO_RANDOM_DEVICE)
config_define_if_not(LIBCXX_ENABLE_LOCALIZATION _LIBCPP_HAS_NO_LOCALIZATION)
config_define_if_not(LIBCXX_ENABLE_UNICODE _LIBCPP_HAS_NO_UNICODE)
config_define_if_not(LIBCXX_ENABLE_VENDOR_AVAILABILITY_ANNOTATIONS _LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS)
# Incomplete features get their own specific disabling flags. This makes it
# easier to grep for target specific flags once the feature is complete.
config_define_if_not(LIBCXX_ENABLE_INCOMPLETE_FEATURES _LIBCPP_HAS_NO_INCOMPLETE_FORMAT)
config_define_if_not(LIBCXX_ENABLE_INCOMPLETE_FEATURES _LIBCPP_HAS_NO_INCOMPLETE_RANGES)

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

if (WIN32 AND LIBCXX_ENABLE_STATIC_ABI_LIBRARY)
  # If linking libcxxabi statically into libcxx, skip the dllimport attributes
  # on symbols we refer to from libcxxabi.
  add_definitions(-D_LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS)
endif()

# Setup all common build flags =================================================
function(cxx_add_common_build_flags target)
  cxx_add_basic_build_flags(${target})
  cxx_add_warning_flags(${target})
  cxx_add_windows_flags(${target})
  cxx_add_exception_flags(${target})
  cxx_add_rtti_flags(${target})
  cxx_add_module_flags(${target})
  cxx_link_system_libraries(${target})
endfunction()

#===============================================================================
# Setup Source Code And Tests
#===============================================================================
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(utils)

set(LIBCXX_TEST_DEPS "")

if (LIBCXX_ENABLE_EXPERIMENTAL_LIBRARY)
  list(APPEND LIBCXX_TEST_DEPS cxx_experimental)
endif()

if (LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY)
  list(APPEND LIBCXX_TEST_DEPS cxx_external_threads)
endif()

if (LIBCXX_INCLUDE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if (LIBCXX_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(lib/abi)
  if (LIBCXX_STANDALONE_BUILD)
    include(AddLLVM) # for get_llvm_lit_path
    # Make sure the llvm-lit script is generated into the bin directory, and
    # do it after adding all tests, since the generated script will only work
    # correctly discovered tests against test locations from the source tree
    # that have already been discovered.
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit
                     ${CMAKE_CURRENT_BINARY_DIR}/llvm-lit)
  endif()
endif()

if (LIBCXX_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()
