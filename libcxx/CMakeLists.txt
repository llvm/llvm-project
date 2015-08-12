# See www/CMake.html for instructions on how to build libcxx with CMake.

#===============================================================================
# Setup Project
#===============================================================================

project(libcxx CXX C)
cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # Set MACOSX_RPATH=YES by default
endif()

set(PACKAGE_NAME libcxx)
set(PACKAGE_VERSION trunk-svn)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvm-bugs@lists.llvm.org")

# Add path for custom modules
set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  ${CMAKE_MODULE_PATH}
  )

# Require out of source build.
include(MacroEnsureOutOfSourceBuild)
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate
 build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(LIBCXX_LIBDIR_SUFFIX "" CACHE STRING
      "Define suffix of library directory name (32/64)")

  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LIBCXX_LIBDIR_SUFFIX})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LIBCXX_LIBDIR_SUFFIX})

  set(LIBCXX_BUILT_STANDALONE 1)
else()
  set(LIBCXX_LIBDIR_SUFFIX ${LLVM_LIBDIR_SUFFIX})
endif()

#===============================================================================
# Setup CMake Options
#===============================================================================

# Define options.
option(LIBCXX_ENABLE_EXCEPTIONS "Use exceptions." ON)
option(LIBCXX_ENABLE_RTTI "Use run time type information." ON)
option(LIBCXX_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(LIBCXX_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LIBCXX_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(LIBCXX_ENABLE_CXX1Y "Enable -std=c++1y and use of c++1y language features if the compiler supports it." OFF)
option(LIBCXX_ENABLE_SHARED "Build libc++ as a shared library." ON)
option(LIBCXX_ENABLE_GLOBAL_FILESYSTEM_NAMESPACE "Build libc++ with support for the global filesystem namespace." ON)
option(LIBCXX_ENABLE_STDIN "Build libc++ with support for stdin/std::cin." ON)
option(LIBCXX_ENABLE_STDOUT "Build libc++ with support for stdout/std::cout." ON)
option(LIBCXX_ENABLE_THREADS "Build libc++ with support for threads." ON)
option(LIBCXX_ENABLE_THREAD_UNSAFE_C_FUNCTIONS "Build libc++ with support for thread-unsafe C functions" ON)
option(LIBCXX_BUILD_32_BITS "Build 32 bit libc++" OFF)
option(LIBCXX_ENABLE_MONOTONIC_CLOCK
  "Build libc++ with support for a monotonic clock.
   This option may only be used when LIBCXX_ENABLE_THREADS=OFF." ON)
option(LIBCXX_INSTALL_HEADERS "Install the libc++ headers." ON)
option(LIBCXX_INSTALL_SUPPORT_HEADERS "Install libc++ support headers." ON)
option(LIBCXX_GENERATE_COVERAGE "Enable generating code coverage." OFF)
set(LIBCXX_COVERAGE_LIBRARY "" CACHE STRING
       "The Profile-rt library used to build with code coverage")
option(LIBCXX_ENABLE_STATIC_ABI_LIBRARY "Statically link the ABI library" OFF)
set(LIBCXX_SYSROOT "" CACHE STRING "Use alternate sysroot.")
set(LIBCXX_GCC_TOOLCHAIN "" CACHE STRING "Use alternate GCC toolchain.")
if (LIBCXX_BUILT_STANDALONE)
  set(LLVM_USE_SANITIZER "" CACHE STRING
      "Define the sanitizer used to build the library and tests")
endif()

if (LIBCXX_ENABLE_STATIC_ABI_LIBRARY)
  if (APPLE)
    message(FATAL_ERROR "LIBCXX_ENABLE_STATIC_ABI_LIBRARY is not supported on OS X")
  else()
    message(WARNING "LIBCXX_ENABLE_STATIC_ABI_LIBRARY is an experimental option")
  endif()
endif()

set(CXXABIS none libcxxabi libcxxrt libstdc++ libsupc++)
if (NOT LIBCXX_CXX_ABI)
  if (NOT DEFINED LIBCXX_BUILT_STANDALONE AND
      IS_DIRECTORY "${CMAKE_SOURCE_DIR}/projects/libcxxabi")
    set(LIBCXX_CXX_ABI_LIBNAME "libcxxabi")
    set(LIBCXX_CXX_ABI_INCLUDE_PATHS "${CMAKE_SOURCE_DIR}/projects/libcxxabi/include")
    set(LIBCXX_CXX_ABI_INTREE 1)
  else ()
    set(LIBCXX_CXX_ABI_LIBNAME "none")
  endif ()
else ()
  set(LIBCXX_CXX_ABI_LIBNAME "${LIBCXX_CXX_ABI}")
endif ()
set(LIBCXX_CXX_ABI "${LIBCXX_CXX_ABI}" CACHE STRING
    "Specify C++ ABI library to use." FORCE)
set_property(CACHE LIBCXX_CXX_ABI PROPERTY STRINGS ;${CXXABIS})

# Build libc++abi with libunwind. We need this option to determine whether to
# link with libunwind or libgcc_s while running the test cases.
option(LIBCXXABI_USE_LLVM_UNWINDER "Build and use the LLVM unwinder." OFF)

#===============================================================================
# Configure System
#===============================================================================

set(LIBCXX_COMPILER    ${CMAKE_CXX_COMPILER})
set(LIBCXX_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBCXX_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})
set(LIBCXX_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LIBCXX_LIBDIR_SUFFIX})

# Declare libc++ configuration variables.
# They are intended for use as follows:
# LIBCXX_CXX_FLAGS: General flags for both the compiler and linker.
# LIBCXX_COMPILE_FLAGS: Compile only flags.
# LIBCXX_LINK_FLAGS: Linker only flags.
set(LIBCXX_CXX_FLAGS "")
set(LIBCXX_COMPILE_FLAGS "")
set(LIBCXX_LINK_FLAGS "")

# Configure compiler.
include(config-ix)
# Configure ABI library
include(HandleLibCXXABI)
# Configure coverage options.
if (LIBCXX_GENERATE_COVERAGE)
  include(CodeCoverage)
  set(CMAKE_BUILD_TYPE "COVERAGE" CACHE STRING "" FORCE)
endif()

#===============================================================================
# Setup Compiler Flags
#===============================================================================

# Get required flags.
# On all systems the system c++ standard library headers need to be excluded.
if (MSVC)
  # MSVC only has -X, which disables all default includes; including the crt.
  # Thus, we do nothing and hope we don't accidentally include any of the C++
  # headers.
else()
  if (LIBCXX_HAS_NOSTDINCXX_FLAG)
    list(APPEND LIBCXX_COMPILE_FLAGS -nostdinc++)
    string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-stdlib=libstdc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  endif()
  # If c++1y has been enabled then attempt to use it. Fail if it is no supported
  # by the compiler. Otherwise choose c++11 and ensure the compiler supports it.
  if (LIBCXX_ENABLE_CXX1Y)
    if (LIBCXX_HAS_STDCXX1Y_FLAG)
      set(LIBCXX_STD_VERSION c++1y)
    else()
      message(FATAL_ERROR "c++1y was enabled but the compiler does not support it.")
    endif()
  else()
    if (LIBCXX_HAS_STDCXX11_FLAG)
      set(LIBCXX_STD_VERSION c++11)
    else()
      message(FATAL_ERROR "c++11 is required by libc++ but is not supported by the compiler")
    endif()
  endif()
  # LIBCXX_STD_VERSION should always be set at this point.
  list(APPEND LIBCXX_CXX_FLAGS "-std=${LIBCXX_STD_VERSION}")
endif()

macro(append_if list condition var)
  if (${condition})
    list(APPEND ${list} ${var})
  endif()
endmacro()

# Get warning flags
if (NOT MSVC)
  append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_WALL_FLAG -Wall)
  list(APPEND LIBCXX_COMPILE_FLAGS -Werror=return-type)
endif()

append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_W_FLAG -W)
append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_WNO_UNUSED_PARAMETER_FLAG -Wno-unused-parameter)
append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_WWRITE_STRINGS_FLAG -Wwrite-strings)
append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_WNO_LONG_LONG_FLAG -Wno-long-long)
if (LIBCXX_ENABLE_WERROR)
  append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_WERROR_FLAG -Werror)
  append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_WX_FLAG -WX)
else()
  append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_WNO_ERROR_FLAG -Wno-error)
  append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_NO_WX_FLAG -WX-)
endif()
if (LIBCXX_ENABLE_PEDANTIC)
  append_if(LIBCXX_COMPILE_FLAGS LIBCXX_HAS_PEDANTIC_FLAG -pedantic)
endif()

# Get feature flags.
# Exceptions
if (LIBCXX_ENABLE_EXCEPTIONS)
  # Catches C++ exceptions only and tells the compiler to assume that extern C
  # functions never throw a C++ exception.
  append_if(LIBCXX_CXX_FLAGS LIBCXX_HAS_EHSC_FLAG -EHsc)
else()
  list(APPEND LIBCXX_CXX_FLAGS -D_LIBCPP_NO_EXCEPTIONS)
  append_if(LIBCXX_CXX_FLAGS LIBCXX_HAS_NO_EHS_FLAG -EHs-)
  append_if(LIBCXX_CXX_FLAGS LIBCXX_HAS_NO_EHA_FLAG -EHa-)
  append_if(LIBCXX_CXX_FLAGS LIBCXX_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions)
endif()
# RTTI
if (NOT LIBCXX_ENABLE_RTTI)
  list(APPEND LIBCXX_CXX_FLAGS -D_LIBCPP_NO_RTTI)
  append_if(LIBCXX_CXX_FLAGS LIBCXX_HAS_NO_GR_FLAG -GR-)
  append_if(LIBCXX_CXX_FLAGS LIBCXX_HAS_FNO_RTTI_FLAG -fno-rtti)
endif()
# Assert
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (LIBCXX_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if (NOT MSVC)
    list(APPEND LIBCXX_COMPILE_FLAGS -D_DEBUG)
  endif()
  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXX_COMPILE_FLAGS -UNDEBUG)
  endif()
else()
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXX_COMPILE_FLAGS -DNDEBUG)
  endif()
endif()
# Static library
if (NOT LIBCXX_ENABLE_SHARED)
  list(APPEND LIBCXX_COMPILE_FLAGS -D_LIBCPP_BUILD_STATIC)
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32)
  if (LIBCXX_BUILD_32_BITS)
    message(STATUS "Building 32 bits executables and libraries.")
    list(APPEND LIBCXX_CXX_FLAGS "-m32")
  endif()
elseif(LIBCXX_BUILD_32_BITS)
  message(FATAL_ERROR "LIBCXX_BUILD_32_BITS=ON is not supported on this platform.")
endif()
# This is the _ONLY_ place where add_definitions is called.
if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# LIBCXX_ENABLE_GLOBAL_FILESYSTEM_NAMESPACE configuration
if (NOT LIBCXX_ENABLE_GLOBAL_FILESYSTEM_NAMESPACE)
  add_definitions(-D_LIBCPP_HAS_NO_GLOBAL_FILESYSTEM_NAMESPACE)
endif()

# LIBCXX_ENABLE_STDIN configuration
if (NOT LIBCXX_ENABLE_STDIN)
  add_definitions(-D_LIBCPP_HAS_NO_STDIN)
endif()

# LIBCXX_ENABLE_STDOUT configuration
if (NOT LIBCXX_ENABLE_STDOUT)
  add_definitions(-D_LIBCPP_HAS_NO_STDOUT)
endif()

# LIBCXX_ENABLE_THREADS configuration
if (NOT LIBCXX_ENABLE_THREADS)
  add_definitions(-D_LIBCPP_HAS_NO_THREADS)
  if (NOT LIBCXX_ENABLE_MONOTONIC_CLOCK)
    add_definitions(-D_LIBCPP_HAS_NO_MONOTONIC_CLOCK)
  endif()
# Ensure LIBCXX_ENABLE_MONOTONIC_CLOCK is set to ON.
elseif(NOT LIBCXX_ENABLE_MONOTONIC_CLOCK)
  message(FATAL_ERROR "LIBCXX_ENABLE_MONOTONIC_CLOCK can only be set to OFF"
                      " when LIBCXX_ENABLE_THREADS is also set to OFF.")
endif()

# LIBCXX_ENABLE_THREAD_UNSAFE_C_FUNCTIONS configuration
if (NOT LIBCXX_ENABLE_THREAD_UNSAFE_C_FUNCTIONS)
  add_definitions(-D_LIBCPP_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS)
endif()

# Configure for sanitizers. If LIBCXX_BUILT_STANDALONE then we have to do
# the flag translation ourselves. Othewise LLVM's CMakeList.txt will handle it.
if (LIBCXX_BUILT_STANDALONE)
  # NOTE: LLVM_USE_SANITIZER checks for a UNIX like system instead of MSVC.
  # But we don't have LLVM_ON_UNIX so checking for MSVC is the best we can do.
  if (LLVM_USE_SANITIZER AND NOT MSVC)
    append_if(LIBCXX_CXX_FLAGS LIBCXX_HAS_FNO_OMIT_FRAME_POINTER_FLAG
              "-fno-omit-frame-pointer")
    if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND
        NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
      append_if(LIBCXX_CXX_FLAGS LIBCXX_HAS_GLINE_TABLES_ONLY_FLAG
                "-gline-tables-only")
    endif()
    if (LLVM_USE_SANITIZER STREQUAL "Address")
      list(APPEND LIBCXX_CXX_FLAGS "-fsanitize=address")
    elseif (LLVM_USE_SANITIZER MATCHES "Memory(WithOrigins)?")
      list(APPEND LIBCXX_CXX_FLAGS "-fsanitize=memory")
      if (LLVM_USE_SANITIZER STREQUAL "MemoryWithOrigins")
        list(APPEND LIBCXX_CXX_FLAGS "-fsanitize-memory-track-origins")
      endif()
    elseif (LLVM_USE_SANITIZER STREQUAL "Undefined")
      list(APPEND LIBCXX_CXX_FLAGS
          "-fsanitize=undefined -fno-sanitize=vptr,function -fno-sanitize-recover")
    elseif (LLVM_USE_SANITIZER STREQUAL "Thread")
      list(APPEND LIBCXX_CXX_FLAGS "-fsanitize=thread")
    else()
      message(WARNING "Unsupported value of LLVM_USE_SANITIZER: ${LLVM_USE_SANITIZER}")
    endif()
  elseif(MSVC)
    message(WARNING "LLVM_USE_SANITIZER is not supported with MSVC")
  endif()
endif()

append_if(LIBCXX_CXX_FLAGS LIBCXX_TARGET_TRIPLE
          "-target ${LIBCXX_TARGET_TRIPLE}")

append_if(LIBCXX_CXX_FLAGS LIBCXX_SYSROOT "--sysroot ${LIBCXX_SYSROOT}")
append_if(LIBCXX_CXX_FLAGS LIBCXX_GCC_TOOLCHAIN
          "-gcc-toolchain ${LIBCXX_GCC_TOOLCHAIN}")

if (LLVM_USE_SANITIZER AND LIBCXX_GENERATE_COVERAGE)
  message(FATAL_ERROR "LLVM_USE_SANITIZER cannot be used with LIBCXX_GENERATE_COVERAGE")
endif()

string(REPLACE ";" " " LIBCXX_CXX_FLAGS "${LIBCXX_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBCXX_CXX_FLAGS}")

#===============================================================================
# Setup Source Code
#===============================================================================

include_directories(include)
add_subdirectory(include)

# Add source code. This also contains all of the logic for deciding linker flags
# soname, etc...
add_subdirectory(lib)

#===============================================================================
# Setup Tests
#===============================================================================

add_subdirectory(test)
