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

set(LIBUNWIND_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBUNWIND_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(LIBUNWIND_LIBCXX_PATH "${CMAKE_CURRENT_LIST_DIR}/../libcxx" CACHE PATH
        "Specify path to libc++ source.")

include(GNUInstallDirs)

#===============================================================================
# Setup CMake Options
#===============================================================================
include(CMakeDependentOption)
include(HandleCompilerRT)

# Define options.
option(LIBUNWIND_BUILD_32_BITS "Build 32 bit multilib libunwind. This option is not supported anymore when building the runtimes. Please specify a full triple instead." ${LLVM_BUILD_32_BITS})
if (LIBUNWIND_BUILD_32_BITS)
  message(FATAL_ERROR "LIBUNWIND_BUILD_32_BITS is not supported anymore when building the runtimes, please specify a full triple instead.")
endif()

option(LIBUNWIND_ENABLE_CET "Build libunwind with CET enabled." OFF)
option(LIBUNWIND_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(LIBUNWIND_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LIBUNWIND_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(LIBUNWIND_ENABLE_SHARED "Build libunwind as a shared library." ON)
option(LIBUNWIND_ENABLE_STATIC "Build libunwind as a static library." ON)
option(LIBUNWIND_ENABLE_CROSS_UNWINDING "Enable cross-platform unwinding support." OFF)
option(LIBUNWIND_ENABLE_ARM_WMMX "Enable unwinding support for ARM WMMX registers." OFF)
option(LIBUNWIND_ENABLE_THREADS "Build libunwind with threading support." ON)
option(LIBUNWIND_WEAK_PTHREAD_LIB "Use weak references to refer to pthread functions." OFF)
option(LIBUNWIND_USE_COMPILER_RT "Use compiler-rt instead of libgcc" OFF)
option(LIBUNWIND_INCLUDE_DOCS "Build the libunwind documentation." ${LLVM_INCLUDE_DOCS})
option(LIBUNWIND_INCLUDE_TESTS "Build the libunwind tests." ${LLVM_INCLUDE_TESTS})
option(LIBUNWIND_IS_BAREMETAL "Build libunwind for baremetal targets." OFF)
option(LIBUNWIND_USE_FRAME_HEADER_CACHE "Cache frame headers for unwinding. Requires locking dl_iterate_phdr." OFF)
option(LIBUNWIND_REMEMBER_HEAP_ALLOC "Use heap instead of the stack for .cfi_remember_state." OFF)
option(LIBUNWIND_INSTALL_HEADERS "Install the libunwind headers." ON)

set(LIBUNWIND_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}" CACHE STRING
    "Define suffix of library directory name (32/64)")
option(LIBUNWIND_INSTALL_LIBRARY "Install the libunwind library." ON)
cmake_dependent_option(LIBUNWIND_INSTALL_STATIC_LIBRARY
  "Install the static libunwind library." ON
  "LIBUNWIND_ENABLE_STATIC;LIBUNWIND_INSTALL_LIBRARY" OFF)
cmake_dependent_option(LIBUNWIND_INSTALL_SHARED_LIBRARY
  "Install the shared libunwind library." ON
  "LIBUNWIND_ENABLE_SHARED;LIBUNWIND_INSTALL_LIBRARY" OFF)

# TODO: Remove this after branching for LLVM 15
if(LIBUNWIND_SYSROOT OR LIBUNWIND_TARGET_TRIPLE OR LIBUNWIND_GCC_TOOLCHAIN)
  message(WARNING "LIBUNWIND_SYSROOT, LIBUNWIND_TARGET_TRIPLE and LIBUNWIND_GCC_TOOLCHAIN are not supported anymore, please use the native CMake equivalents instead")
endif()

if (LIBUNWIND_ENABLE_SHARED)
  set(LIBUNWIND_DEFAULT_TEST_CONFIG "llvm-libunwind-shared.cfg.in")
else()
  set(LIBUNWIND_DEFAULT_TEST_CONFIG "llvm-libunwind-static.cfg.in")
endif()
set(LIBUNWIND_TEST_CONFIG "${LIBUNWIND_DEFAULT_TEST_CONFIG}" CACHE STRING
  "The path to the Lit testing configuration to use when running the tests.
   If a relative path is provided, it is assumed to be relative to '<monorepo>/libunwind/test/configs'.")
if (NOT IS_ABSOLUTE "${LIBUNWIND_TEST_CONFIG}")
  set(LIBUNWIND_TEST_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/test/configs/${LIBUNWIND_TEST_CONFIG}")
endif()
message(STATUS "Using libunwind testing configuration: ${LIBUNWIND_TEST_CONFIG}")
set(LIBUNWIND_TEST_PARAMS "" CACHE STRING
    "A list of parameters to run the Lit test suite with.")

if (NOT LIBUNWIND_ENABLE_SHARED AND NOT LIBUNWIND_ENABLE_STATIC)
  message(FATAL_ERROR "libunwind must be built as either a shared or static library.")
endif()

if (LIBUNWIND_ENABLE_CET AND MSVC)
  message(FATAL_ERROR "libunwind CET support is not available for MSVC!")
endif()

if (WIN32)
  set(LIBUNWIND_DEFAULT_HIDE_SYMBOLS TRUE)
else()
  set(LIBUNWIND_DEFAULT_HIDE_SYMBOLS FALSE)
endif()
option(LIBUNWIND_HIDE_SYMBOLS
  "Do not export any symbols from the static library." ${LIBUNWIND_DEFAULT_HIDE_SYMBOLS})

#===============================================================================
# Configure System
#===============================================================================

# Add path for custom modules
set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    ${CMAKE_MODULE_PATH})

set(LIBUNWIND_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}" CACHE PATH
    "Path where built libunwind headers should be installed.")
set(LIBUNWIND_INSTALL_RUNTIME_DIR "${CMAKE_INSTALL_BINDIR}" CACHE PATH
    "Path where built libunwind runtime libraries should be installed.")

set(LIBUNWIND_SHARED_OUTPUT_NAME "unwind" CACHE STRING "Output name for the shared libunwind runtime library.")
set(LIBUNWIND_STATIC_OUTPUT_NAME "unwind" CACHE STRING "Output name for the static libunwind runtime library.")

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(LIBUNWIND_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE})
  set(LIBUNWIND_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE} CACHE PATH
      "Path where built libunwind libraries should be installed.")
  if(LIBCXX_LIBDIR_SUBDIR)
    string(APPEND LIBUNWIND_LIBRARY_DIR /${LIBUNWIND_LIBDIR_SUBDIR})
    string(APPEND LIBUNWIND_INSTALL_LIBRARY_DIR /${LIBUNWIND_LIBDIR_SUBDIR})
  endif()
else()
  if(LLVM_LIBRARY_OUTPUT_INTDIR)
    set(LIBUNWIND_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  else()
    set(LIBUNWIND_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBUNWIND_LIBDIR_SUFFIX})
  endif()
  set(LIBUNWIND_INSTALL_LIBRARY_DIR lib${LIBUNWIND_LIBDIR_SUFFIX} CACHE PATH
      "Path where built libunwind libraries should be installed.")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBUNWIND_LIBRARY_DIR})

set(LIBUNWIND_C_FLAGS "")
set(LIBUNWIND_CXX_FLAGS "")
set(LIBUNWIND_COMPILE_FLAGS "")
set(LIBUNWIND_LINK_FLAGS "")

# Include macros for adding and removing libunwind flags.
include(HandleLibunwindFlags)

#===============================================================================
# Setup Compiler Flags
#===============================================================================

# Configure compiler.
include(config-ix)

if (LIBUNWIND_USE_COMPILER_RT AND NOT LIBUNWIND_HAS_NODEFAULTLIBS_FLAG)
  list(APPEND LIBUNWIND_LINK_FLAGS "-rtlib=compiler-rt")
endif()

add_compile_flags_if_supported(-Werror=return-type)

if (LIBUNWIND_ENABLE_CET)
  add_compile_flags_if_supported(-fcf-protection=full)
  add_compile_flags_if_supported(-mshstk)
  if (NOT CXX_SUPPORTS_FCF_PROTECTION_EQ_FULL_FLAG)
    message(SEND_ERROR "Compiler doesn't support CET -fcf-protection option!")
  endif()
  if (NOT CXX_SUPPORTS_MSHSTK_FLAG)
    message(SEND_ERROR "Compiler doesn't support CET -mshstk option!")
  endif()
endif()

# Get warning flags
add_compile_flags_if_supported(-W)
add_compile_flags_if_supported(-Wall)
add_compile_flags_if_supported(-Wchar-subscripts)
add_compile_flags_if_supported(-Wconversion)
add_compile_flags_if_supported(-Wmismatched-tags)
add_compile_flags_if_supported(-Wmissing-braces)
add_compile_flags_if_supported(-Wnewline-eof)
add_compile_flags_if_supported(-Wno-unused-function)
add_compile_flags_if_supported(-Wshadow)
add_compile_flags_if_supported(-Wshorten-64-to-32)
add_compile_flags_if_supported(-Wsign-compare)
add_compile_flags_if_supported(-Wsign-conversion)
add_compile_flags_if_supported(-Wstrict-aliasing=2)
add_compile_flags_if_supported(-Wstrict-overflow=4)
add_compile_flags_if_supported(-Wunused-parameter)
add_compile_flags_if_supported(-Wunused-variable)
add_compile_flags_if_supported(-Wwrite-strings)
add_compile_flags_if_supported(-Wundef)

add_compile_flags_if_supported(-Wno-suggest-override)

if (WIN32)
  # The headers lack matching dllexport attributes (_LIBUNWIND_EXPORT);
  # silence the warning instead of cluttering the headers (which aren't
  # necessarily the ones that the callers will use anyway) with the
  # attributes.
  add_compile_flags_if_supported(-Wno-dll-attribute-on-redeclaration)
endif()

if (LIBUNWIND_ENABLE_WERROR)
  add_compile_flags_if_supported(-Werror)
  add_compile_flags_if_supported(-WX)
else()
  add_compile_flags_if_supported(-Wno-error)
  add_compile_flags_if_supported(-WX-)
endif()

if (LIBUNWIND_ENABLE_PEDANTIC)
  add_compile_flags_if_supported(-pedantic)
endif()

# Get feature flags.
# Exceptions
# Catches C++ exceptions only and tells the compiler to assume that extern C
# functions never throw a C++ exception.
add_cxx_compile_flags_if_supported(-fstrict-aliasing)
add_cxx_compile_flags_if_supported(-EHsc)

# Don't run the linker in this CMake check.
#
# The reason why this was added is that when building libunwind for
# ARM Linux, we need to pass the -funwind-tables flag in order for it to
# work properly with ARM EHABI.
#
# However, when performing CMake checks, adding this flag causes the check
# to produce a false negative, because the compiler generates calls
# to __aeabi_unwind_cpp_pr0, which is defined in libunwind itself,
# which isn't built yet, so the linker complains about undefined symbols.
#
# This leads to libunwind not being built with this flag, which makes
# libunwind quite useless in this setup.
set(_previous_CMAKE_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
add_compile_flags_if_supported(-funwind-tables)
set(CMAKE_TRY_COMPILE_TARGET_TYPE ${_previous_CMAKE_TRY_COMPILE_TARGET_TYPE})

if (LIBUNWIND_USES_ARM_EHABI AND NOT CXX_SUPPORTS_FUNWIND_TABLES_FLAG)
  message(SEND_ERROR "The -funwind-tables flag must be supported "
                     "because this target uses ARM Exception Handling ABI")
endif()

add_cxx_compile_flags_if_supported(-fno-exceptions)
add_cxx_compile_flags_if_supported(-fno-rtti)

# Ensure that we don't depend on C++ standard library.
if (CXX_SUPPORTS_NOSTDINCXX_FLAG)
  list(APPEND LIBUNWIND_COMPILE_FLAGS -nostdinc++)
  # Remove -stdlib flags to prevent them from causing an unused flag warning.
  string(REPLACE "--stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "--stdlib=libstdc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-stdlib=libstdc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Assert
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (LIBUNWIND_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if (NOT MSVC)
    add_compile_flags(-D_DEBUG)
  endif()

  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_compile_flags(-UNDEBUG)
  endif()
else()
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_compile_flags(-DNDEBUG)
  endif()
endif()

# Cross-unwinding
if (NOT LIBUNWIND_ENABLE_CROSS_UNWINDING)
  add_compile_flags(-D_LIBUNWIND_IS_NATIVE_ONLY)
endif()

# Threading-support
if (NOT LIBUNWIND_ENABLE_THREADS)
  add_compile_flags(-D_LIBUNWIND_HAS_NO_THREADS)
endif()

# ARM WMMX register support
if (LIBUNWIND_ENABLE_ARM_WMMX)
  # __ARM_WMMX is a compiler pre-define (as per the ACLE 2.0). Clang does not
  # define this macro for any supported target at present. Therefore, here we
  # provide the option to explicitly enable support for WMMX registers in the
  # unwinder.
  add_compile_flags(-D__ARM_WMMX)
endif()

if(LIBUNWIND_IS_BAREMETAL)
  add_compile_definitions(_LIBUNWIND_IS_BAREMETAL)
endif()

if(LIBUNWIND_USE_FRAME_HEADER_CACHE)
  add_compile_definitions(_LIBUNWIND_USE_FRAME_HEADER_CACHE)
endif()

if(LIBUNWIND_REMEMBER_HEAP_ALLOC)
  add_compile_definitions(_LIBUNWIND_REMEMBER_HEAP_ALLOC)
endif()

# This is the _ONLY_ place where add_definitions is called.
if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if (C_SUPPORTS_COMMENT_LIB_PRAGMA)
  if (LIBUNWIND_HAS_DL_LIB)
    add_definitions(-D_LIBUNWIND_LINK_DL_LIB)
  endif()
  if (LIBUNWIND_HAS_PTHREAD_LIB)
    add_definitions(-D_LIBUNWIND_LINK_PTHREAD_LIB)
  endif()
endif()

#===============================================================================
# Setup Source Code
#===============================================================================

add_subdirectory(include)

add_subdirectory(src)

if (LIBUNWIND_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

if (LIBUNWIND_INCLUDE_TESTS AND EXISTS ${LLVM_CMAKE_DIR})
  add_subdirectory(test)
endif()
