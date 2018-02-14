# See www/CMake.html for instructions on how to build libcxxabi with CMake.

#===============================================================================
# Setup Project
#===============================================================================

cmake_minimum_required(VERSION 3.4.3)

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # Set MACOSX_RPATH=YES by default
endif()

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(libcxxabi CXX C)

  set(PACKAGE_NAME libcxxabi)
  set(PACKAGE_VERSION 6.0.0)
  set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
  set(PACKAGE_BUGREPORT "llvm-bugs@lists.llvm.org")

  # Find the LLVM sources and simulate LLVM CMake options.
  include(HandleOutOfTreeLLVM)
endif()

# Require out of source build.
include(MacroEnsureOutOfSourceBuild)
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate
 build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )

#===============================================================================
# Setup CMake Options
#===============================================================================
include(CMakeDependentOption)
include(HandleCompilerRT)

# Define options.
option(LIBCXXABI_ENABLE_EXCEPTIONS "Use exceptions." ON)
option(LIBCXXABI_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(LIBCXXABI_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LIBCXXABI_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(LIBCXXABI_USE_LLVM_UNWINDER "Build and use the LLVM unwinder." OFF)
option(LIBCXXABI_ENABLE_STATIC_UNWINDER "Statically link the LLVM unwinder." OFF)
option(LIBCXXABI_USE_COMPILER_RT "Use compiler-rt instead of libgcc" OFF)
option(LIBCXXABI_ENABLE_THREADS "Build with threads enabled" ON)
option(LIBCXXABI_HAS_PTHREAD_API "Ignore auto-detection and force use of pthread API" OFF)
option(LIBCXXABI_HAS_EXTERNAL_THREAD_API
  "Build libc++abi with an externalized threading API.
  This option may only be set to ON when LIBCXXABI_ENABLE_THREADS=ON." OFF)
option(LIBCXXABI_BUILD_EXTERNAL_THREAD_LIBRARY
  "Build libc++abi with an externalized threading library.
   This option may only be set to ON when LIBCXXABI_ENABLE_THREADS=ON" OFF)

# FIXME: This option should default to off. Unfortunatly GCC 4.9 fails to link
# programs to due undefined references to new/delete in libc++abi. Once this
# has been fixed or worked around the default value should be changed.
option(LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS
    "Build libc++abi with definitions for operator new/delete. Normally libc++
    provides these definitions" ON)
option(LIBCXXABI_BUILD_32_BITS "Build 32 bit libc++abi." ${LLVM_BUILD_32_BITS})
option(LIBCXXABI_INCLUDE_TESTS "Generate build targets for the libc++abi unit tests." ${LLVM_INCLUDE_TESTS})
set(LIBCXXABI_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}" CACHE STRING
    "Define suffix of library directory name (32/64)")
option(LIBCXXABI_INSTALL_LIBRARY "Install the libc++abi library." ON)
set(LIBCXXABI_TARGET_TRIPLE "" CACHE STRING "Target triple for cross compiling.")
set(LIBCXXABI_GCC_TOOLCHAIN "" CACHE PATH "GCC toolchain for cross compiling.")
set(LIBCXXABI_SYSROOT "" CACHE PATH "Sysroot for cross compiling.")
set(LIBCXXABI_LIBCXX_LIBRARY_PATH "" CACHE PATH "The path to libc++ library.")

# Default to building a shared library so that the default options still test
# the libc++abi that is being built. There are two problems with testing a
# static libc++abi. In the case of a standalone build, the tests will link the
# system's libc++, which might not have been built against our libc++abi. In the
# case of an in tree build, libc++ will prefer a dynamic libc++abi from the
# system over a static libc++abi from the output directory.
option(LIBCXXABI_ENABLE_SHARED "Build libc++abi as a shared library." ON)
option(LIBCXXABI_ENABLE_STATIC "Build libc++abi as a static library." ON)

option(LIBCXXABI_BAREMETAL "Build libc++abi for baremetal targets." OFF)
# The default terminate handler attempts to demangle uncaught exceptions, which
# causes extra I/O and demangling code to be pulled in.
option(LIBCXXABI_SILENT_TERMINATE "Set this to make the terminate handler default to a silent alternative" OFF)

if (NOT LIBCXXABI_ENABLE_SHARED AND NOT LIBCXXABI_ENABLE_STATIC)
  message(FATAL_ERROR "libc++abi must be built as either a shared or static library.")
endif()

if (LLVM_EXTERNAL_LIBCXX_SOURCE_DIR)
  set(LIBCXXABI_LIBCXX_SRC_DIRS ${LLVM_EXTERNAL_LIBCXX_SOURCE_DIR})
else()
  set(LIBCXXABI_LIBCXX_SRC_DIRS
    "${LLVM_MAIN_SRC_DIR}/projects/libcxx"
    "${LLVM_MAIN_SRC_DIR}/runtimes/libcxx"
    )
endif()

set(LIBCXXABI_LIBCXX_INCLUDE_DIRS "")
foreach(dir ${LIBCXXABI_LIBCXX_SRC_DIRS})
  list(APPEND LIBCXXABI_LIBCXX_INCLUDE_DIRS "${dir}/include")
endforeach()

find_path(
  LIBCXXABI_LIBCXX_INCLUDES
  vector
  PATHS ${LIBCXXABI_LIBCXX_INCLUDES}
        ${LIBCXXABI_LIBCXX_PATH}/include
        ${CMAKE_BINARY_DIR}/${LIBCXXABI_LIBCXX_INCLUDES}
        ${LIBCXXABI_LIBCXX_INCLUDE_DIRS}
        ${LLVM_INCLUDE_DIR}/c++/v1
  )

set(LIBCXXABI_LIBCXX_INCLUDES "${LIBCXXABI_LIBCXX_INCLUDES}" CACHE PATH
    "Specify path to libc++ includes." FORCE)

find_path(
  LIBCXXABI_LIBCXX_PATH
  utils/libcxx/test/__init__.py
  PATHS ${LIBCXXABI_LIBCXX_PATH}
        ${LIBCXXABI_LIBCXX_INCLUDES}/../
        ${LIBCXXABI_LIBCXX_SRC_DIRS}
  NO_DEFAULT_PATH
  )

if (LIBCXXABI_LIBCXX_PATH STREQUAL "LIBCXXABI_LIBCXX_PATH-NOTFOUND")
  message(WARNING "LIBCXXABI_LIBCXX_PATH was not specified and couldn't be infered.")
  set(LIBCXXABI_LIBCXX_PATH "")
endif()

set(LIBCXXABI_LIBCXX_PATH "${LIBCXXABI_LIBCXX_PATH}" CACHE PATH
    "Specify path to libc++ source." FORCE)

#===============================================================================
# Configure System
#===============================================================================

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )

set(LIBCXXABI_COMPILER    ${CMAKE_CXX_COMPILER})
set(LIBCXXABI_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBCXXABI_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
if (LLVM_LIBRARY_OUTPUT_INTDIR)
  set(LIBCXXABI_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
else()
  set(LIBCXXABI_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBCXXABI_LIBDIR_SUFFIX})
endif()

set(LIBCXXABI_INSTALL_PREFIX "" CACHE STRING
    "Define libc++abi destination prefix.")

if (NOT LIBCXXABI_INSTALL_PREFIX MATCHES "^$|.*/")
  message(FATAL_ERROR "LIBCXXABI_INSTALL_PREFIX has to end with \"/\".")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBCXXABI_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBCXXABI_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBCXXABI_LIBRARY_DIR})

# By default, for non-standalone builds, libcxx and libcxxabi share a library
# directory.
if (NOT LIBCXXABI_LIBCXX_LIBRARY_PATH)
  set(LIBCXXABI_LIBCXX_LIBRARY_PATH "${LIBCXXABI_LIBRARY_DIR}" CACHE PATH
      "The path to libc++ library.")
endif()

# Check that we can build with 32 bits if requested.
if (CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32)
  if (LIBCXXABI_BUILD_32_BITS AND NOT LLVM_BUILD_32_BITS) # Don't duplicate the output from LLVM
    message(STATUS "Building 32 bits executables and libraries.")
  endif()
elseif(LIBCXXABI_BUILD_32_BITS)
  message(FATAL_ERROR "LIBCXXABI_BUILD_32_BITS=ON is not supported on this platform.")
endif()

# Declare libc++abi configuration variables.
# They are intended for use as follows:
# LIBCXXABI_C_FLAGS: General flags for both the c++ compiler and linker.
# LIBCXXABI_CXX_FLAGS: General flags for both the c++ compiler and linker.
# LIBCXXABI_COMPILE_FLAGS: Compile only flags.
# LIBCXXABI_LINK_FLAGS: Linker only flags.
# LIBCXXABI_LIBRARIES: libraries libc++abi is linked to.

set(LIBCXXABI_C_FLAGS "")
set(LIBCXXABI_CXX_FLAGS "")
set(LIBCXXABI_COMPILE_FLAGS "")
set(LIBCXXABI_LINK_FLAGS "")
set(LIBCXXABI_LIBRARIES "")

# Include macros for adding and removing libc++abi flags.
include(HandleLibcxxabiFlags)

#===============================================================================
# Setup Compiler Flags
#===============================================================================

# Configure target flags
add_target_flags_if(LIBCXXABI_BUILD_32_BITS "-m32")
add_target_flags_if(LIBCXXABI_TARGET_TRIPLE
          "--target=${LIBCXXABI_TARGET_TRIPLE}")
add_target_flags_if(LIBCXXABI_GCC_TOOLCHAIN
         "--gcc-toolchain=${LIBCXXABI_GCC_TOOLCHAIN}")
add_target_flags_if(LIBCXXABI_SYSROOT
          "--sysroot=${LIBCXXABI_SYSROOT}")

if (LIBCXXABI_TARGET_TRIPLE)
  set(TARGET_TRIPLE "${LIBCXXABI_TARGET_TRIPLE}")
endif()

# Configure compiler. Must happen after setting the target flags.
include(config-ix)

if (LIBCXXABI_HAS_NOSTDINCXX_FLAG)
  list(APPEND LIBCXXABI_COMPILE_FLAGS -nostdinc++)
  # Remove -stdlib flags to prevent them from causing an unused flag warning.
  string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-stdlib=libstdc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if (LIBCXXABI_USE_COMPILER_RT)
  list(APPEND LIBCXXABI_LINK_FLAGS "-rtlib=compiler-rt")
endif()

# Let the library headers know they are currently being used to build the
# library.
add_definitions(-D_LIBCXXABI_BUILDING_LIBRARY)

# Disable DLL annotations on Windows for static builds.
if (WIN32 AND LIBCXXABI_ENABLE_STATIC AND NOT LIBCXXABI_ENABLE_SHARED)
  add_definitions(-D_LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS)
endif()

add_compile_flags_if_supported(-Werror=return-type)

# Get warning flags
add_compile_flags_if_supported(-W)
add_compile_flags_if_supported(-Wall)
add_compile_flags_if_supported(-Wchar-subscripts)
add_compile_flags_if_supported(-Wconversion)
add_compile_flags_if_supported(-Wmismatched-tags)
add_compile_flags_if_supported(-Wmissing-braces)
add_compile_flags_if_supported(-Wnewline-eof)
add_compile_flags_if_supported(-Wunused-function)
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

if (LIBCXXABI_ENABLE_WERROR)
  add_compile_flags_if_supported(-Werror)
  add_compile_flags_if_supported(-WX)
else()
  add_compile_flags_if_supported(-Wno-error)
  add_compile_flags_if_supported(-WX-)
endif()
if (LIBCXXABI_ENABLE_PEDANTIC)
  add_compile_flags_if_supported(-pedantic)
endif()

# Get feature flags.
add_compile_flags_if_supported(-fstrict-aliasing)

# Exceptions
if (LIBCXXABI_ENABLE_EXCEPTIONS)
  # Catches C++ exceptions only and tells the compiler to assume that extern C
  # functions never throw a C++ exception.
  add_compile_flags_if_supported(-EHsc)
  # Do we really need to be run through the C compiler ?
  add_c_compile_flags_if_supported(-funwind-tables)
else()
  add_definitions(-D_LIBCXXABI_NO_EXCEPTIONS)
  add_compile_flags_if_supported(-fno-exceptions)
  add_compile_flags_if_supported(-EHs-)
  add_compile_flags_if_supported(-EHa-)
endif()

# Assert
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (LIBCXXABI_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if (NOT MSVC)
    list(APPEND LIBCXXABI_COMPILE_FLAGS -D_DEBUG)
  endif()
  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXXABI_COMPILE_FLAGS -UNDEBUG)
  endif()
else()
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXXABI_COMPILE_FLAGS -DNDEBUG)
  endif()
endif()
# Static library
if (NOT LIBCXXABI_ENABLE_SHARED)
  list(APPEND LIBCXXABI_COMPILE_FLAGS -D_LIBCPP_BUILD_STATIC)
endif()

# Threading
if (NOT LIBCXXABI_ENABLE_THREADS)
  if (LIBCXXABI_HAS_PTHREAD_API)
    message(FATAL_ERROR "LIBCXXABI_HAS_PTHREAD_API can only"
                        " be set to ON when LIBCXXABI_ENABLE_THREADS"
                        " is also set to ON.")
  endif()
  if (LIBCXXABI_HAS_EXTERNAL_THREAD_API)
    message(FATAL_ERROR "LIBCXXABI_HAS_EXTERNAL_THREAD_API can only"
                        " be set to ON when LIBCXXABI_ENABLE_THREADS"
                        " is also set to ON.")
  endif()
  if (LIBCXXABI_BUILD_EXTERNAL_THREAD_LIBRARY)
    message(FATAL_ERROR "LIBCXXABI_BUILD_EXTERNAL_THREAD_LIBRARY can only"
                        " be set to ON when LIBCXXABI_ENABLE_THREADS"
                        " is also set to ON.")
  endif()
  add_definitions(-D_LIBCXXABI_HAS_NO_THREADS)
endif()

if (LIBCXXABI_HAS_EXTERNAL_THREAD_API)
  if (LIBCXXABI_HAS_PTHREAD_API)
    message(FATAL_ERROR "The options LIBCXXABI_HAS_EXTERNAL_THREAD_API"
                        " and LIBCXXABI_HAS_PTHREAD_API cannot be both"
                        " set to ON at the same time.")
  endif()
  if (LIBCXXABI_BUILD_EXTERNAL_THREAD_LIBRARY)
    message(FATAL_ERROR "The options LIBCXXABI_BUILD_EXTERNAL_THREAD_LIBRARY"
                        " and LIBCXXABI_HAS_EXTERNAL_THREAD_API cannot be both"
                        " set to ON at the same time.")
  endif()
endif()

if (LLVM_ENABLE_MODULES)
  # Ignore that the rest of the modules flags are now unused.
  add_compile_flags_if_supported(-Wno-unused-command-line-argument)
  add_compile_flags(-fno-modules)
endif()

set(LIBCXXABI_HAS_UNDEFINED_SYMBOLS OFF)
if ((NOT LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS)
    OR (LIBCXXABI_BUILD_EXTERNAL_THREAD_LIBRARY AND LIBCXXABI_ENABLE_SHARED)
    OR MINGW)
  set(LIBCXXABI_HAS_UNDEFINED_SYMBOLS ON)
endif()

if (LIBCXXABI_HAS_UNDEFINED_SYMBOLS)
  # Need to allow unresolved symbols if this is to work with shared library builds
  if (APPLE)
    list(APPEND LIBCXXABI_LINK_FLAGS "-undefined dynamic_lookup")
  else()
    # Relax this restriction from HandleLLVMOptions
    string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  endif()
endif()

if (LIBCXXABI_HAS_PTHREAD_API)
  add_definitions(-D_LIBCPP_HAS_THREAD_API_PTHREAD)
endif()

if (LIBCXXABI_HAS_EXTERNAL_THREAD_API)
  add_definitions(-D_LIBCPP_HAS_THREAD_API_EXTERNAL)
endif()

if (LIBCXXABI_BUILD_EXTERNAL_THREAD_LIBRARY)
  add_definitions(-D_LIBCPP_HAS_THREAD_LIBRARY_EXTERNAL)
endif()

# Prevent libc++abi from having library dependencies on libc++
add_definitions(-D_LIBCPP_DISABLE_EXTERN_TEMPLATE)

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Define LIBCXXABI_USE_LLVM_UNWINDER for conditional compilation.
if (LIBCXXABI_USE_LLVM_UNWINDER)
  add_definitions(-DLIBCXXABI_USE_LLVM_UNWINDER)
endif()

if (LIBCXXABI_SILENT_TERMINATE)
  add_definitions(-DLIBCXXABI_SILENT_TERMINATE)
endif()

if (LIBCXXABI_BAREMETAL)
    add_definitions(-DLIBCXXABI_BAREMETAL)
endif()

string(REPLACE ";" " " LIBCXXABI_CXX_FLAGS "${LIBCXXABI_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBCXXABI_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${LIBCXXABI_C_FLAGS}")

#===============================================================================
# Setup Source Code
#===============================================================================

set(LIBCXXABI_LIBUNWIND_INCLUDES "${LIBCXXABI_LIBUNWIND_INCLUDES}" CACHE PATH
    "Specify path to libunwind includes." FORCE)
set(LIBCXXABI_LIBUNWIND_PATH "${LIBCXXABI_LIBUNWIND_PATH}" CACHE PATH
    "Specify path to libunwind source." FORCE)

include_directories(include)
if (LIBCXXABI_USE_LLVM_UNWINDER OR LLVM_NATIVE_ARCH MATCHES ARM)
  find_path(
    LIBCXXABI_LIBUNWIND_INCLUDES_INTERNAL
    libunwind.h
    PATHS ${LIBCXXABI_LIBUNWIND_INCLUDES}
          ${LIBCXXABI_LIBUNWIND_PATH}/include
          ${CMAKE_BINARY_DIR}/${LIBCXXABI_LIBUNWIND_INCLUDES}
          ${LLVM_MAIN_SRC_DIR}/projects/libunwind/include
          ${LLVM_MAIN_SRC_DIR}/runtimes/libunwind/include
    NO_DEFAULT_PATH
  )

  if (LIBCXXABI_LIBUNWIND_INCLUDES_INTERNAL STREQUAL "LIBCXXABI_LIBUNWIND_INCLUDES_INTERNAL-NOTFOUND")
    set(LIBCXXABI_LIBUNWIND_INCLUDES_INTERNAL "")
  endif()

  if (NOT LIBCXXABI_LIBUNWIND_INCLUDES_INTERNAL STREQUAL "")
    include_directories("${LIBCXXABI_LIBUNWIND_INCLUDES_INTERNAL}")
  endif()
endif()

# Add source code. This also contains all of the logic for deciding linker flags
# soname, etc...
add_subdirectory(src)

if (NOT LIBCXXABI_INCLUDE_TESTS OR (LIBCXXABI_STANDALONE_BUILD AND NOT LIBCXXABI_ENABLE_SHARED))
  # We can't reasonably test the system C++ library with a static libc++abi.
  # We either need to be able to replace libc++abi at run time (with a shared
  # libc++abi), or we need to be able to replace the C++ runtime (with a non-
  # standalone build).
  message(WARNING "The libc++abi tests aren't valid when libc++abi is built "
                  "standalone (i.e. outside of llvm/projects/libcxxabi ) and "
                  "is built without a shared library.  Either build a shared "
                  "library, build libc++abi at the same time as you build "
                  "libc++, or do without testing.  No check target will be "
                  "available!")
else()
  add_subdirectory(test)
  add_subdirectory(fuzz)
endif()
