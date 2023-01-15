# See https://libcxx.llvm.org/docs/BuildingLibcxx.html for instructions on how
# to build libcxx with CMake.

#===============================================================================
# Setup Project
#===============================================================================
cmake_minimum_required(VERSION 3.13.4)

set(LLVM_COMMON_CMAKE_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

# Add path for custom modules
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  "${LLVM_COMMON_CMAKE_UTILS}"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

set(CMAKE_FOLDER "libc++")

set(LIBCXX_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBCXX_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(LIBCXX_BINARY_INCLUDE_DIR "${LIBCXX_BINARY_DIR}/include/c++build")

include(GNUInstallDirs)

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
option(LIBCXX_ENABLE_ASSERTIONS
  "Enable assertions inside the compiled library, and at the same time make it the
   default when compiling user code. Note that assertions can be enabled or disabled
   by users in their own code regardless of this option." OFF)
option(LIBCXX_ENABLE_SHARED "Build libc++ as a shared library." ON)
option(LIBCXX_ENABLE_STATIC "Build libc++ as a static library." ON)
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
option(LIBCXX_ENABLE_DEBUG_MODE
  "Whether to build libc++ with the debug mode enabled.
   By default, this is turned off. Turning it on results in a different ABI (additional
   symbols but also potentially different layouts of types), and one should not mix code
   built against a dylib that has debug mode and code built against a regular dylib." OFF)
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
option(LIBCXX_ENABLE_FSTREAM
   "Whether to include support for <fstream>." ON) # TODO: Consider rolling that into LIBCXX_ENABLE_FILESYSTEM
option(LIBCXX_ENABLE_UNICODE
  "Whether to include support for Unicode in the library. Disabling Unicode can
   be useful when porting to platforms that don't support UTF-8 encoding (e.g.
   embedded)." ON)
option(LIBCXX_ENABLE_WIDE_CHARACTERS
  "Whether to include support for wide characters in the library. Disabling
   wide character support can be useful when porting to platforms that don't
   support the C functionality for wide characters. When wide characters are
   not supported, several parts of the library will be disabled, notably the
   wide character specializations of std::basic_string." ON)
option(LIBCXX_ENABLE_VENDOR_AVAILABILITY_ANNOTATIONS
  "Whether to turn on vendor availability annotations on declarations that depend
   on definitions in a shared library. By default, we assume that we're not building
   libc++ for any specific vendor, and we disable those annotations. Vendors wishing
   to provide compile-time errors when using features unavailable on some version of
   the shared library they shipped should turn this on and see `include/__availability`
   for more details." OFF)
option(LIBCXX_ENABLE_CLANG_TIDY "Whether to compile and run clang-tidy checks" OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(LIBCXX_DEFAULT_TEST_CONFIG "llvm-libc++-shared-gcc.cfg.in")
elseif(MINGW)
  set(LIBCXX_DEFAULT_TEST_CONFIG "llvm-libc++-mingw.cfg.in")
elseif(WIN32) # clang-cl
  if (LIBCXX_ENABLE_SHARED)
    set(LIBCXX_DEFAULT_TEST_CONFIG "llvm-libc++-shared-clangcl.cfg.in")
  else()
    set(LIBCXX_DEFAULT_TEST_CONFIG "llvm-libc++-static-clangcl.cfg.in")
  endif()
else()
  if (LIBCXX_ENABLE_SHARED)
    set(LIBCXX_DEFAULT_TEST_CONFIG "llvm-libc++-shared.cfg.in")
  else()
    set(LIBCXX_DEFAULT_TEST_CONFIG "llvm-libc++-static.cfg.in")
  endif()
endif()
set(LIBCXX_TEST_CONFIG "${LIBCXX_DEFAULT_TEST_CONFIG}" CACHE STRING
    "The path to the Lit testing configuration to use when running the tests.
     If a relative path is provided, it is assumed to be relative to '<monorepo>/libcxx/test/configs'.")
if (NOT IS_ABSOLUTE "${LIBCXX_TEST_CONFIG}")
  set(LIBCXX_TEST_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/test/configs/${LIBCXX_TEST_CONFIG}")
endif()
message(STATUS "Using libc++ testing configuration: ${LIBCXX_TEST_CONFIG}")
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

option(LIBCXX_ABI_UNSTABLE "Use the unstable ABI of libc++. This is equivalent to specifying LIBCXX_ABI_VERSION=n, where n is the not-yet-stable version." OFF)
if (LIBCXX_ABI_UNSTABLE)
  set(abi_version "2")
else()
  set(abi_version "1")
endif()
set(LIBCXX_ABI_VERSION "${abi_version}" CACHE STRING
  "ABI version of libc++. Can be either 1 or 2, where 2 is currently the unstable ABI.
   Defaults to 1 unless LIBCXX_ABI_UNSTABLE is specified, in which case this is 2.")
set(LIBCXX_LIBRARY_VERSION "${LIBCXX_ABI_VERSION}.0" CACHE STRING
  "Version of libc++. This will be reflected in the name of the shared library produced.
   For example, -DLIBCXX_LIBRARY_VERSION=x.y will result in the library being named
   libc++.x.y.dylib, along with the usual symlinks pointing to that. On Apple platforms,
   this also controls the linker's 'current_version' property.")
set(LIBCXX_ABI_NAMESPACE "__${LIBCXX_ABI_VERSION}" CACHE STRING "The inline ABI namespace used by libc++. It defaults to __n where `n` is the current ABI version.")
if (NOT LIBCXX_ABI_NAMESPACE MATCHES "__.*")
  message(FATAL_ERROR "LIBCXX_ABI_NAMESPACE must be a reserved identifier, got '${LIBCXX_ABI_NAMESPACE}'.")
endif()
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

set(LIBCXX_ABI_DEFINES "" CACHE STRING "A semicolon separated list of ABI macros to define in the site config header.")
option(LIBCXX_EXTRA_SITE_DEFINES "Extra defines to add into __config_site")
option(LIBCXX_USE_COMPILER_RT "Use compiler-rt instead of libgcc" OFF)

option(LIBCXX_ENABLE_BACKWARDS_COMPATIBILITY_DEBUG_MODE_SYMBOLS
  "Whether to include the old Debug mode symbols in the compiled library. This
   is provided for backwards compatibility since the compiled library used to
   always contain those symbols, regardless of whether the library was built
   with the debug mode enabled. This is OFF by default, please contact the libc++
   developers if you need to turn this on, as this will be removed in LLVM 16." OFF)

# ABI Library options ---------------------------------------------------------
if (LIBCXX_TARGETING_MSVC)
  set(LIBCXX_DEFAULT_ABI_LIBRARY "vcruntime")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "FreeBSD")
  set(LIBCXX_DEFAULT_ABI_LIBRARY "libcxxrt")
else()
  set(LIBCXX_DEFAULT_ABI_LIBRARY "libcxxabi")
endif()

set(LIBCXX_SUPPORTED_ABI_LIBRARIES none libcxxabi system-libcxxabi libcxxrt libstdc++ libsupc++ vcruntime)
set(LIBCXX_CXX_ABI "${LIBCXX_DEFAULT_ABI_LIBRARY}" CACHE STRING "Specify C++ ABI library to use. Supported values are ${LIBCXX_SUPPORTED_ABI_LIBRARIES}.")
if (NOT "${LIBCXX_CXX_ABI}" IN_LIST LIBCXX_SUPPORTED_ABI_LIBRARIES)
  message(FATAL_ERROR "Unsupported C++ ABI library: '${LIBCXX_CXX_ABI}'. Supported values are ${LIBCXX_SUPPORTED_ABI_LIBRARIES}.")
endif()

option(LIBCXX_ENABLE_STATIC_ABI_LIBRARY
  "Use a static copy of the ABI library when linking libc++.
   This option cannot be used with LIBCXX_ENABLE_ABI_LINKER_SCRIPT." OFF)

option(LIBCXX_STATICALLY_LINK_ABI_IN_STATIC_LIBRARY
  "Statically link the ABI library to static library"
  ${LIBCXX_ENABLE_STATIC_ABI_LIBRARY})

option(LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY
  "Statically link the ABI library to shared library"
  ${LIBCXX_ENABLE_STATIC_ABI_LIBRARY})

# Generate and install a linker script inplace of libc++.so. The linker script
# will link libc++ to the correct ABI library. This option is on by default
# on UNIX platforms other than Apple unless
# 'LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY' is on. This option is also
# disabled when the ABI library is not specified or is specified to be "none".
set(ENABLE_LINKER_SCRIPT_DEFAULT_VALUE OFF)
if (LLVM_HAVE_LINK_VERSION_SCRIPT AND NOT LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY
      AND NOT LIBCXX_CXX_ABI STREQUAL "none"
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
option(LIBCXX_BUILD_32_BITS "Build 32 bit multilib libc++. This option is not supported anymore when building the runtimes. Please specify a full triple instead." ${LLVM_BUILD_32_BITS})
if (LIBCXX_BUILD_32_BITS)
  message(FATAL_ERROR "LIBCXX_BUILD_32_BITS is not supported anymore when building the runtimes, please specify a full triple instead.")
endif()

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

set(LIBCXX_HERMETIC_STATIC_LIBRARY_DEFAULT OFF)
if (WIN32)
  set(LIBCXX_HERMETIC_STATIC_LIBRARY_DEFAULT ON)
endif()
option(LIBCXX_HERMETIC_STATIC_LIBRARY
  "Do not export any symbols from the static library." ${LIBCXX_HERMETIC_STATIC_LIBRARY_DEFAULT})

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
        LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY cannot be specified with
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

set(LIBCXX_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/c++/v1" CACHE PATH
    "Path where target-agnostic libc++ headers should be installed.")
set(LIBCXX_INSTALL_RUNTIME_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path where built libc++ runtime libraries should be installed.")

set(LIBCXX_SHARED_OUTPUT_NAME "c++" CACHE STRING "Output name for the shared libc++ runtime library.")
set(LIBCXX_STATIC_OUTPUT_NAME "c++" CACHE STRING "Output name for the static libc++ runtime library.")

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(LIBCXX_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE})
  set(LIBCXX_GENERATED_INCLUDE_DIR "${LLVM_BINARY_DIR}/include/c++/v1")
  set(LIBCXX_GENERATED_INCLUDE_TARGET_DIR "${LLVM_BINARY_DIR}/include/${LLVM_DEFAULT_TARGET_TRIPLE}/c++/v1")
  set(LIBCXX_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE} CACHE PATH
      "Path where built libc++ libraries should be installed.")
  set(LIBCXX_INSTALL_INCLUDE_TARGET_DIR "${CMAKE_INSTALL_INCLUDEDIR}/${LLVM_DEFAULT_TARGET_TRIPLE}/c++/v1" CACHE PATH
      "Path where target-specific libc++ headers should be installed.")
  if(LIBCXX_LIBDIR_SUBDIR)
    string(APPEND LIBCXX_LIBRARY_DIR /${LIBCXX_LIBDIR_SUBDIR})
    string(APPEND LIBCXX_INSTALL_LIBRARY_DIR /${LIBCXX_LIBDIR_SUBDIR})
  endif()
else()
  if(LLVM_LIBRARY_OUTPUT_INTDIR)
    set(LIBCXX_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
    set(LIBCXX_GENERATED_INCLUDE_DIR "${LLVM_BINARY_DIR}/include/c++/v1")
  else()
    set(LIBCXX_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBCXX_LIBDIR_SUFFIX})
    set(LIBCXX_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include/c++/v1")
  endif()
  set(LIBCXX_GENERATED_INCLUDE_TARGET_DIR "${LIBCXX_GENERATED_INCLUDE_DIR}")
  set(LIBCXX_INSTALL_LIBRARY_DIR lib${LIBCXX_LIBDIR_SUFFIX} CACHE PATH
      "Path where built libc++ libraries should be installed.")
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
set(LIBCXX_ADDITIONAL_COMPILE_FLAGS "" CACHE STRING
    "Additional Compile only flags which can be provided in cache")
set(LIBCXX_ADDITIONAL_LIBRARIES "" CACHE STRING
    "Additional libraries libc++ is linked to which can be provided in cache")

# Include macros for adding and removing libc++ flags.
include(HandleLibcxxFlags)

# Target flags ================================================================
# These flags get added to CMAKE_CXX_FLAGS and CMAKE_C_FLAGS so that
# 'config-ix' use them during feature checks. It also adds them to both
# 'LIBCXX_COMPILE_FLAGS' and 'LIBCXX_LINK_FLAGS'

if (${CMAKE_SYSTEM_NAME} MATCHES "AIX")
  add_target_flags_if_supported("-mdefault-visibility-export-mapping=explicit")
  set(CMAKE_AIX_EXPORT_ALL_SYMBOLS OFF)
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
  # because they are provided by the dylib itself with the exception of z/OS.
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

  if (C_SUPPORTS_COMMENT_LIB_PRAGMA)
    if (LIBCXX_HAS_PTHREAD_LIB)
      target_compile_definitions(${target} PRIVATE -D_LIBCPP_LINK_PTHREAD_LIB)
    endif()
    if (LIBCXX_HAS_RT_LIB)
      target_compile_definitions(${target} PRIVATE -D_LIBCPP_LINK_RT_LIB)
    endif()
  endif()
  target_compile_options(${target} PUBLIC "${LIBCXX_ADDITIONAL_COMPILE_FLAGS}")
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
                                                          -Werror=return-type -Wextra-semi -Wundef
                                                          -Wformat-nonliteral)
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
      -Wno-attributes
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
    elseif (USE_SANITIZER STREQUAL "HWAddress")
      append_flags(SANITIZER_FLAGS "-fsanitize=hwaddress")
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

get_sanitizer_flags(SANITIZER_FLAGS "${LLVM_USE_SANITIZER}")

# Link system libraries =======================================================
function(cxx_link_system_libraries target)

# In order to remove just libc++ from the link step
# we need to use -nostdlib++ whenever it is supported.
# Unfortunately this cannot be used universally because for example g++ supports
# only -nodefaultlibs in which case all libraries will be removed and
# all libraries but c++ have to be added in manually.
  if (CXX_SUPPORTS_NOSTDLIBXX_FLAG)
    target_add_link_flags_if_supported(${target} PRIVATE "-nostdlib++")
  else()
    target_add_link_flags_if_supported(${target} PRIVATE "-nodefaultlibs")
    target_add_compile_flags_if_supported(${target} PRIVATE "/Zl")
    target_add_link_flags_if_supported(${target} PRIVATE "/nodefaultlib")
  endif()

  if (CXX_SUPPORTS_UNWINDLIB_EQ_NONE_FLAG AND LIBCXXABI_USE_LLVM_UNWINDER)
    # If we're linking directly against the libunwind that we're building
    # in the same invocation, don't try to link in the toolchain's
    # default libunwind (which may be missing still).
    target_add_link_flags_if_supported(${target} PRIVATE "--unwindlib=none")
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
  target_link_libraries(${target} PUBLIC "${LIBCXX_ADDITIONAL_LIBRARIES}")
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
    # Clang-cl shared builds don't support the experimental library.
    # To avoid linker errors the format_error destructor is inlined for the
    # dylib. Users can never use format in this mode.
    # TODO FMT Remove when format becomes mainline.
    if (LIBCXX_ENABLE_SHARED)
      target_compile_definitions(${target} PRIVATE
                                 _LIBCPP_INLINE_FORMAT_ERROR_DTOR)
    endif()
  endif()
endfunction()

# Configuration file flags =====================================================
config_define(${LIBCXX_ABI_VERSION} _LIBCPP_ABI_VERSION)
config_define(${LIBCXX_ABI_NAMESPACE} _LIBCPP_ABI_NAMESPACE)
config_define_if(LIBCXX_ABI_FORCE_ITANIUM _LIBCPP_ABI_FORCE_ITANIUM)
config_define_if(LIBCXX_ABI_FORCE_MICROSOFT _LIBCPP_ABI_FORCE_MICROSOFT)
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
config_define_if_not(LIBCXX_ENABLE_FSTREAM _LIBCPP_HAS_NO_FSTREAM)
config_define_if_not(LIBCXX_ENABLE_UNICODE _LIBCPP_HAS_NO_UNICODE)
config_define_if_not(LIBCXX_ENABLE_WIDE_CHARACTERS _LIBCPP_HAS_NO_WIDE_CHARACTERS)
config_define_if_not(LIBCXX_ENABLE_VENDOR_AVAILABILITY_ANNOTATIONS _LIBCPP_HAS_NO_VENDOR_AVAILABILITY_ANNOTATIONS)
config_define_if(LIBCXX_ENABLE_DEBUG_MODE _LIBCPP_ENABLE_DEBUG_MODE)
if (LIBCXX_ENABLE_ASSERTIONS)
  config_define(1 _LIBCPP_ENABLE_ASSERTIONS_DEFAULT)
else()
  config_define(0 _LIBCPP_ENABLE_ASSERTIONS_DEFAULT)
endif()

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

if (LIBCXX_EXTRA_SITE_DEFINES)
  set(extra_site_defines)
  foreach (extra_site_define ${LIBCXX_EXTRA_SITE_DEFINES})
    # Allow defines such as DEFINE=VAL, transformed into "#define DEFINE VAL".
    string(REPLACE "=" " " extra_site_define "${extra_site_define}")
    list(APPEND extra_site_defines "#define ${extra_site_define}")
  endforeach()
  string(REPLACE ";" "\n" extra_site_defines "${extra_site_defines}")
  config_define(${extra_site_defines} _LIBCPP_EXTRA_SITE_DEFINES)
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

set(LIBCXX_TEST_DEPS "cxx_experimental")

if (LIBCXX_BUILD_EXTERNAL_THREAD_LIBRARY)
  list(APPEND LIBCXX_TEST_DEPS cxx_external_threads)
endif()

if (LIBCXX_ENABLE_CLANG_TIDY)
  list(APPEND LIBCXX_TEST_DEPS cxx-tidy)
endif()

if (LIBCXX_INCLUDE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if (LIBCXX_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(lib/abi)
endif()

if (LIBCXX_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()
