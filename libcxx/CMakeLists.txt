# See www/CMake.html for instructions on how to build libcxx with CMake.

#===============================================================================
# Setup Project
#===============================================================================

project(libcxx CXX C)
cmake_minimum_required(VERSION 2.8)

set(PACKAGE_NAME libcxx)
set(PACKAGE_VERSION trunk-svn)
set(PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set(PACKAGE_BUGREPORT "llvmbugs@cs.uiuc.edu")

# Add path for custom modules
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
  )

# Require out of source build.
include(MacroEnsureOutOfSourceBuild)
MACRO_ENSURE_OUT_OF_SOURCE_BUILD(
 "${PROJECT_NAME} requires an out of source build. Please create a separate
 build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there."
 )

#===============================================================================
# Setup CMake Options
#===============================================================================

# Define options.
option(LIBCXX_ENABLE_EXCEPTIONS "Use exceptions." ON)
option(LIBCXX_ENABLE_RTTI "Use run time type information." ON)
option(LIBCXX_ENABLE_ASSERTIONS "Enable assertions independent of build mode." ON)
option(LIBCXX_ENABLE_PEDANTIC "Compile with pedantic enabled." ON)
option(LIBCXX_ENABLE_WERROR "Fail and stop if a warning is triggered." OFF)
option(LIBCXX_ENABLE_CXX0X "Enable -std=c++0x and use of c++0x language features if the compiler supports it." ON)
option(LIBCXX_ENABLE_SHARED "Build libc++ as a shared library." ON)

#===============================================================================
# Configure System
#===============================================================================

# Get triples.
include(GetTriple)
get_host_triple(LIBCXX_HOST_TRIPLE
  LIBCXX_HOST_ARCH
  LIBCXX_HOST_VENDOR
  LIBCXX_HOST_OS
  )
set(LIBCXX_HOST_TRIPLE ${LIBCXX_HOST_TRIPLE} CACHE STRING "Host triple.")
get_target_triple(LIBCXX_TARGET_TRIPLE
  LIBCXX_TARGET_ARCH
  LIBCXX_TARGET_VENDOR
  LIBCXX_TARGET_OS
  )
set(LIBCXX_TARGET_TRIPLE ${LIBCXX_TARGET_TRIPLE} CACHE STRING "Target triple.")

# Configure compiler.
include(config-ix)

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
    set(LIBCXX_CXX_REQUIRED_FLAGS -nostdinc++)
  endif()
  if (LIBCXX_ENABLE_CXX0X AND LIBCXX_HAS_STDCXX0X_FLAG)
    list(APPEND LIBCXX_CXX_REQUIRED_FLAGS -std=c++0x)
  endif()
endif()

macro(append_if list condition var)
  if (${condition})
    list(APPEND ${list} ${var})
  endif()
endmacro()

# Get warning flags
append_if(LIBCXX_WARNING_FLAGS LIBCXX_HAS_WALL_FLAG -Wall)
append_if(LIBCXX_WARNING_FLAGS LIBCXX_HAS_W_FLAG -W)
append_if(LIBCXX_WARNING_FLAGS LIBCXX_HAS_WNO_UNUSED_PARAMETER_FLAG -Wno-unused-parameter)
append_if(LIBCXX_WARNING_FLAGS LIBCXX_HAS_WWRITE_STRINGS_FLAG -Wwrite-strings)
append_if(LIBCXX_WARNING_FLAGS LIBCXX_HAS_WNO_LONG_LONG_FLAG -Wno-long-long)
if (LIBCXX_ENABLE_WERROR)
  append_if(LIBCXX_WARNING_FLAGS LIBCXX_HAS_WERROR_FLAG -Werror)
  append_if(LIBCXX_WARNING_FLAGS LIBCXX_HAS_WX_FLAG -WX)
endif()
if (LIBCXX_ENABLE_PEDANTIC)
  append_if(LIBCXX_WARNING_FLAGS LIBCXX_HAS_PEDANTIC_FLAG -pedantic)
endif()

# Get feature flags.
# Exceptions
if (LIBCXX_ENABLE_EXCEPTIONS)
  # Catches C++ exceptions only and tells the compiler to assume that extern C
  # functions never throw a C++ exception.
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_EHSC_FLAG -EHsc)
else()
  list(APPEND LIBCXX_CXX_FEATURE_FLAGS -D_LIBCPP_NO_EXCEPTIONS)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_NO_EHS_FLAG -EHs-)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_NO_EHA_FLAG -EHa-)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_FNO_EXCEPTIONS_FLAG -fno-exceptions)
endif()
# RTTI
if (NOT LIBCXX_ENABLE_RTTI)
  list(APPEND LIBCXX_CXX_FEATURE_FLAGS -D_LIBCPP_NO_RTTI)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_NO_GR_FLAG -GR-)
  append_if(LIBCXX_CXX_FEATURE_FLAGS LIBCXX_HAS_FNO_RTTI_FLAG -fno-rtti)
endif()
# Assert
if (LLVM_ENABLE_ASSERTIONS)
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if (NOT MSVC)
    list(APPEND LIBCXX_CXX_FEATURE_FLAGS -D_DEBUG)
  endif()
  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXX_CXX_FEATURE_FLAGS -UNDEBUG)
  endif()
else()
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    list(APPEND LIBCXX_CXX_FEATURE_FLAGS -DNDEBUG)
  endif()
endif()

# This is the _ONLY_ place where add_definitions is called.
add_definitions(
 ${LIBCXX_CXX_REQUIRED_FLAGS}
 ${LIBCXX_CXX_WARNING_FLAGS}
 ${LIBCXX_CXX_FEATURE_FLAGS}
 )

#===============================================================================
# Setup Source Code
#===============================================================================

include_directories(include)

# Add source code. This also contains all of the logic for deciding linker flags
# soname, etc...
add_subdirectory(lib)

#===============================================================================
# Setup Tests
#===============================================================================

add_subdirectory(test)
