# See www/CMake.html for instructions on how to build libcxxabi with CMake.

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../libcxx")
  message(FATAL_ERROR "libc++abi now requires being built in a monorepo layout with libcxx available")
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

set(LIBCXXABI_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBCXXABI_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(LIBCXXABI_LIBCXX_PATH "${CMAKE_CURRENT_LIST_DIR}/../libcxx" CACHE PATH
        "Specify path to libc++ source.")

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR LIBCXXABI_STANDALONE_BUILD)
  project(libcxxabi CXX C)

  set(PACKAGE_NAME libcxxabi)
  set(PACKAGE_VERSION 11.0.0)
  set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
  set(PACKAGE_BUGREPORT "llvm-bugs@lists.llvm.org")

  # Add the CMake module path of libcxx so we can reuse HandleOutOfTreeLLVM.cmake
  set(LIBCXXABI_LIBCXX_CMAKE_PATH "${LIBCXXABI_LIBCXX_PATH}/cmake/Modules")
  list(APPEND CMAKE_MODULE_PATH "${LIBCXXABI_LIBCXX_CMAKE_PATH}")

  # In a standalone build, we don't have llvm to automatically generate the
  # llvm-lit script for us.  So we need to provide an explicit directory that
  # the configurator should write the script into.
  set(LIBCXXABI_STANDALONE_BUILD 1)
  set(LLVM_LIT_OUTPUT_DIR "${LIBCXXABI_BINARY_DIR}/bin")

  # Find the LLVM sources and simulate LLVM CMake options.
  include(HandleOutOfTreeLLVM)
endif()

if (LIBCXXABI_STANDALONE_BUILD)
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

#===============================================================================
# Setup CMake Options
#===============================================================================
include(CMakeDependentOption)
include(HandleCompilerRT)

# Define options.
option(LIBCXXABI_ENABLE_EXCEPTIONS
  "Provide support for exceptions in the runtime.
  When disabled, libc++abi does not support stack unwinding and other exceptions-related features." ON)
option(LIBCXXABI_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(LIBCXXABI_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LIBCXXABI_ENABLE_PIC "Build Position-Independent Code, even in static library" ON)
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
option(LIBCXXABI_ENABLE_FORGIVING_DYNAMIC_CAST
"Make dynamic_cast more forgiving when type_info's mistakenly have hidden \
visibility, and thus multiple type_infos can exist for a single type. \
When the dynamic_cast would normally fail, this option will cause the \
library to try comparing the type_info names to see if they are equal \
instead." OFF)

option(LIBCXXABI_ENABLE_NEW_DELETE_DEFINITIONS
  "Build libc++abi with definitions for operator new/delete. These are normally
   defined in libc++abi, but it is also possible to define them in libc++, in
   which case the definition in libc++abi should be turned off." ON)
option(LIBCXXABI_BUILD_32_BITS "Build 32 bit libc++abi." ${LLVM_BUILD_32_BITS})
option(LIBCXXABI_INCLUDE_TESTS "Generate build targets for the libc++abi unit tests." ${LLVM_INCLUDE_TESTS})
set(LIBCXXABI_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}" CACHE STRING
    "Define suffix of library directory name (32/64)")
option(LIBCXXABI_INSTALL_LIBRARY "Install the libc++abi library." ON)
set(LIBCXXABI_TARGET_TRIPLE "" CACHE STRING "Target triple for cross compiling.")
set(LIBCXXABI_GCC_TOOLCHAIN "" CACHE PATH "GCC toolchain for cross compiling.")
set(LIBCXXABI_SYSROOT "" CACHE PATH "Sysroot for cross compiling.")
set(LIBCXXABI_LIBCXX_LIBRARY_PATH "" CACHE PATH "The path to libc++ library.")
set(LIBCXXABI_LIBRARY_VERSION "1.0" CACHE STRING
"Version of libc++abi. This will be reflected in the name of the shared \
library produced. For example, -DLIBCXXABI_LIBRARY_VERSION=x.y will \
result in the library being named libc++abi.x.y.dylib, along with the \
usual symlinks pointing to that.")

# Default to building a shared library so that the default options still test
# the libc++abi that is being built. There are two problems with testing a
# static libc++abi. In the case of a standalone build, the tests will link the
# system's libc++, which might not have been built against our libc++abi. In the
# case of an in tree build, libc++ will prefer a dynamic libc++abi from the
# system over a static libc++abi from the output directory.
option(LIBCXXABI_ENABLE_SHARED "Build libc++abi as a shared library." ON)
option(LIBCXXABI_ENABLE_STATIC "Build libc++abi as a static library." ON)

cmake_dependent_option(LIBCXXABI_INSTALL_STATIC_LIBRARY
  "Install the static libc++abi library." ON
  "LIBCXXABI_ENABLE_STATIC;LIBCXXABI_INSTALL_LIBRARY" OFF)
cmake_dependent_option(LIBCXXABI_INSTALL_SHARED_LIBRARY
  "Install the shared libc++abi library." ON
  "LIBCXXABI_ENABLE_SHARED;LIBCXXABI_INSTALL_LIBRARY" OFF)

cmake_dependent_option(LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_STATIC_LIBRARY
  "Statically link the LLVM unwinder to static library" ON
  "LIBCXXABI_ENABLE_STATIC_UNWINDER;LIBCXXABI_ENABLE_STATIC" OFF)
cmake_dependent_option(LIBCXXABI_STATICALLY_LINK_UNWINDER_IN_SHARED_LIBRARY
  "Statically link the LLVM unwinder to shared library" ON
  "LIBCXXABI_ENABLE_STATIC_UNWINDER;LIBCXXABI_ENABLE_SHARED" OFF)

option(LIBCXXABI_BAREMETAL "Build libc++abi for baremetal targets." OFF)
# The default terminate handler attempts to demangle uncaught exceptions, which
# causes extra I/O and demangling code to be pulled in.
option(LIBCXXABI_SILENT_TERMINATE "Set this to make the terminate handler default to a silent alternative" OFF)
option(LIBCXXABI_NON_DEMANGLING_TERMINATE "Set this to make the terminate handler
avoid demangling" OFF)

if (NOT LIBCXXABI_ENABLE_SHARED AND NOT LIBCXXABI_ENABLE_STATIC)
  message(FATAL_ERROR "libc++abi must be built as either a shared or static library.")
endif()

set(LIBCXXABI_LIBCXX_INCLUDES "${LIBCXXABI_LIBCXX_PATH}/include" CACHE PATH
    "Specify path to libc++ includes.")
message(STATUS "Libc++abi will be using libc++ includes from ${LIBCXXABI_LIBCXX_INCLUDES}")

option(LIBCXXABI_HERMETIC_STATIC_LIBRARY
  "Do not export any symbols from the static library." OFF)

set(LIBCXXABI_TEST_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/test/lit.site.cfg.in" CACHE STRING
    "The Lit testing configuration to use when running the tests.")
set(LIBCXXABI_TEST_PARAMS "" CACHE STRING
    "A list of parameters to run the Lit test suite with.")

#===============================================================================
# Configure System
#===============================================================================

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR AND NOT APPLE)
  set(LIBCXXABI_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR}/${LLVM_DEFAULT_TARGET_TRIPLE}/c++)
  set(LIBCXXABI_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE}/c++)
  if(LIBCXX_LIBDIR_SUBDIR)
    string(APPEND LIBCXXABI_LIBRARY_DIR /${LIBCXXABI_LIBDIR_SUBDIR})
    string(APPEND LIBCXXABI_INSTALL_LIBRARY_DIR /${LIBCXXABI_LIBDIR_SUBDIR})
  endif()
elseif(LLVM_LIBRARY_OUTPUT_INTDIR)
  set(LIBCXXABI_LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  set(LIBCXXABI_INSTALL_LIBRARY_DIR lib${LIBCXXABI_LIBDIR_SUFFIX})
else()
  set(LIBCXXABI_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBCXXABI_LIBDIR_SUFFIX})
  set(LIBCXXABI_INSTALL_LIBRARY_DIR lib${LIBCXXABI_LIBDIR_SUFFIX})
endif()

set(LIBCXXABI_INSTALL_PREFIX "" CACHE STRING "Define libc++abi destination prefix.")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBCXXABI_LIBRARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBCXXABI_LIBRARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LIBCXXABI_LIBRARY_DIR})

# By default, for non-standalone builds, libcxx and libcxxabi share a library
# directory.
if (NOT LIBCXXABI_LIBCXX_LIBRARY_PATH)
  set(LIBCXXABI_LIBCXX_LIBRARY_PATH "${LIBCXXABI_LIBRARY_DIR}" CACHE PATH
      "The path to libc++ library." FORCE)
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

if(LIBCXXABI_TARGET_TRIPLE)
  add_target_flags("--target=${LIBCXXABI_TARGET_TRIPLE}")
elseif(CMAKE_CXX_COMPILER_TARGET)
  set(LIBCXXABI_TARGET_TRIPLE "${CMAKE_CXX_COMPILER_TARGET}")
endif()
if(LIBCXX_GCC_TOOLCHAIN)
  add_target_flags("--gcc-toolchain=${LIBCXXABI_GCC_TOOLCHAIN}")
elseif(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
  set(LIBCXXABI_GCC_TOOLCHAIN "${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}")
endif()
if(LIBCXXABI_SYSROOT)
  add_target_flags("--sysroot=${LIBCXXABI_SYSROOT}")
elseif(CMAKE_SYSROOT)
  set(LIBCXXABI_SYSROOT "${CMAKE_SYSROOT}")
endif()

if (LIBCXXABI_TARGET_TRIPLE)
  set(TARGET_TRIPLE "${LIBCXXABI_TARGET_TRIPLE}")
endif()

# Configure compiler. Must happen after setting the target flags.
include(config-ix)

if (LIBCXXABI_HAS_NOSTDINCXX_FLAG)
  list(APPEND LIBCXXABI_COMPILE_FLAGS -nostdinc++)
  # cmake 3.14 and above remove system include paths that are explicitly
  # passed on the command line.  We build with -nostdinc++ and explicitly add
  # just the libcxx system include paths with -I on the command line.
  # Setting CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES effectively prevents cmake
  # from removing these.
  # See: https://gitlab.kitware.com/cmake/cmake/issues/19227
  set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "")
  # Remove -stdlib flags to prevent them from causing an unused flag warning.
  string(REPLACE "--stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "--stdlib=libstdc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-stdlib=libstdc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Let the library headers know they are currently being used to build the
# library.
add_definitions(-D_LIBCXXABI_BUILDING_LIBRARY)

# libcxxabi needs to, for various reasons, include the libcpp headers as if
# it is being built as part of libcxx.
add_definitions(-D_LIBCPP_BUILDING_LIBRARY)

# Disable DLL annotations on Windows for static builds.
if (WIN32 AND LIBCXXABI_ENABLE_STATIC AND NOT LIBCXXABI_ENABLE_SHARED)
  if (LIBCXX_ENABLE_SHARED AND LIBCXX_ENABLE_STATIC_ABI_LIBRARY)
    # Building libcxxabi statically, but intending for it to be statically
    # linked into a shared libcxx; keep dllexport enabled within libcxxabi,
    # as the symbols will need to be exported from libcxx.
  else()
    # Regular static build; disable dllexports.
    add_definitions(-D_LIBCXXABI_DISABLE_VISIBILITY_ANNOTATIONS)
  endif()
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

add_compile_flags_if_supported(-Wno-suggest-override)

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
  add_definitions(-D_LIBCPP_HAS_NO_THREADS)
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

# Bring back `std::unexpected`, which is removed in C++17, to support
# pre-C++17.
add_definitions(-D_LIBCPP_ENABLE_CXX17_REMOVED_UNEXPECTED_FUNCTIONS)

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

if (LIBCXXABI_NON_DEMANGLING_TERMINATE)
  add_definitions(-DLIBCXXABI_NON_DEMANGLING_TERMINATE)
endif()

if (LIBCXXABI_BAREMETAL)
    add_definitions(-DLIBCXXABI_BAREMETAL)
endif()

if (LIBCXXABI_HAS_COMMENT_LIB_PRAGMA)
  if (LIBCXXABI_HAS_PTHREAD_LIB)
    add_definitions(-D_LIBCXXABI_LINK_PTHREAD_LIB)
  endif()
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
  find_path(LIBCXXABI_LIBUNWIND_INCLUDES_INTERNAL libunwind.h
    PATHS ${LIBCXXABI_LIBUNWIND_INCLUDES}
          ${LIBCXXABI_LIBUNWIND_PATH}/include
          ${CMAKE_BINARY_DIR}/${LIBCXXABI_LIBUNWIND_INCLUDES}
          ${LLVM_MAIN_SRC_DIR}/projects/libunwind/include
          ${LLVM_MAIN_SRC_DIR}/runtimes/libunwind/include
          ${LLVM_MAIN_SRC_DIR}/../libunwind/include
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH
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

if (LIBCXXABI_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(fuzz)
endif()
